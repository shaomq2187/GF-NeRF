//
// Created by ppwang on 2022/7/17.
//

#include "Hash3DAnchored.h"
#include <torch/torch.h>
#include "../Utils/Common.h"

using Tensor = torch::Tensor;

TORCH_LIBRARY(dec_hash3d_anchored, m)
{
  std::cout << "register Hash3DAnchoredInfo" << std::endl;
  m.class_<Hash3DAnchoredInfo>("Hash3DAnchoredInfo").def(torch::init());
}

Hash3DAnchored::Hash3DAnchored(int64_t log2_table_size,int64_t n_volume,double learn_rate) {

  learn_rate_ = learn_rate;
  pool_size_ = (1 << log2_table_size) * N_LEVELS;
  // mlp_hidden_dim_ = mlp_hidden_dim;
  // mlp_out_dim_ = mlp_out_dim;
  // n_hidden_layers_ = n_hidden_layers;

  // Feat pool
  feat_pool_ = (torch::rand({pool_size_, N_CHANNELS}, CUDAFloat) * .2f - 1.f) * 1e-4f;
  feat_pool_.requires_grad_(true);
  CHECK(feat_pool_.is_contiguous());

  n_volumes_ = n_volume;
  // Get prime numbers
  auto is_prim = [](int x) {
    for (int i = 2; i * i <= x; i++) {
      if (x % i == 0) return false;
    }
    return true;
  };

  std::vector<int> prim_selected;
  int min_local_prim = 1 << 28;
  int max_local_prim = 1 << 30;

  for (int i = 0; i < 3 * N_LEVELS * n_volumes_; i++) {
    int val;
    do {
      val = torch::randint(min_local_prim, max_local_prim, {1}, CPUInt).item<int>();
    }
    while (!is_prim(val));
    prim_selected.push_back(val);
  }

  CHECK_EQ(prim_selected.size(), 3 * N_LEVELS * n_volumes_);
  
  prim_pool_ = torch::from_blob(prim_selected.data(), 3 * N_LEVELS * n_volumes_, CPUInt).to(torch::kCUDA);
  prim_pool_ = prim_pool_.reshape({N_LEVELS, n_volumes_, 3}).contiguous();

  if (rand_bias) {
    bias_pool_ = (torch::rand({ N_LEVELS * n_volumes_, 3 }, CUDAFloat) * 1000.f + 100.f).contiguous();
  }
  else {
    bias_pool_ = torch::zeros({ N_LEVELS * n_volumes_, 3 }, CUDAFloat).contiguous();
  }

  // Size of each level & each volume.
  {
    int local_size = pool_size_ / N_LEVELS;
    local_size = (local_size >> 4) << 4;
    feat_local_size_  = torch::full({ N_LEVELS }, local_size, CUDAInt).contiguous();
    feat_local_idx_ = torch::cumsum(feat_local_size_, 0) - local_size;
    feat_local_idx_ = feat_local_idx_.to(torch::kInt32).contiguous();
  }

  // MLP
  // mlp_ = std::make_unique<TCNNWP>( N_LEVELS * N_CHANNELS, mlp_out_dim_, mlp_hidden_dim_, n_hidden_layers_);
}
void Hash3DAnchored::ReleaseResources(){
  std::cout<<"[Hash3DAnchored_CXX] ReleaseResources is called"<<std::endl;
  // delete feat_pool_;
  // delete prim_pool_;
  // delete bias_pool_;
  // delete feat_local_idx_;
  // delete feat_local_size_;
  // delete query_points_;
  // delete query_volume_idx_;
  feat_pool_ = torch::empty({0}, torch::kFloat32);
  prim_pool_ = torch::empty({0}, torch::kFloat32);
  bias_pool_ = torch::empty({0}, torch::kFloat32);
  feat_local_idx_ = torch::empty({0}, torch::kFloat32);
  feat_local_size_ = torch::empty({0}, torch::kFloat32);
  query_points_ = torch::empty({0}, torch::kFloat32);
  query_volume_idx_ = torch::empty({0}, torch::kFloat32);

  c10::cuda::CUDACachingAllocator::emptyCache();
}

Tensor Hash3DAnchored::AnchoredQuery(const Tensor& points, const Tensor& anchors) {
#ifdef PROFILE
  ScopeWatch watch(__func__);
#endif
  // std::cout << "[debug] anchored hash n_volumes:"<<n_volumes_ << std::endl;
  auto info = torch::make_intrusive<Hash3DAnchoredInfo>();

  // query_points_ = ((points + 1.f) * .5f).contiguous();   // [-1, 1] -> [0, 1]
  query_points_ = points.contiguous();
  query_volume_idx_ = anchors.contiguous();
  info->hash3d_ = this;
  Tensor feat = torch::autograd::Hash3DAnchoredFunction::apply(feat_pool_, torch::IValue(info))[0];  // [n_points, n_levels * n_channels];

  return feat;
}

int Hash3DAnchored::LoadStates(const std::vector<Tensor> &states, int idx) {
  std::cout<<"[Hash3DAnchored_CXX] LoadStates"<<std::endl;
  feat_pool_.data().copy_(states[idx++]);
  prim_pool_ = states[idx++].clone().to(torch::kCUDA).contiguous();   // The size may changed.
  bias_pool_.data().copy_(states[idx++]);
  n_volumes_ = states[idx++].item<int>();

  // mlp_->params_.data().copy_(states[idx++]);

  return idx;
}

std::vector<Tensor> Hash3DAnchored::States() {
  std::vector<Tensor> ret;
  ret.push_back(feat_pool_.data());
  ret.push_back(prim_pool_.data());
  ret.push_back(bias_pool_.data());
  ret.push_back(torch::full({1}, n_volumes_, CPUInt));

  // ret.push_back(mlp_->params_.data());

  return ret;
}

std::vector<torch::optim::OptimizerParamGroup> Hash3DAnchored::OptimParamGroups() {
  std::vector<torch::optim::OptimizerParamGroup> ret;


  float lr = learn_rate_;
  {
    auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
    opt->betas() = {0.9, 0.99};
    opt->eps() = 1e-15;

    std::vector<Tensor> params = { feat_pool_ };
    ret.emplace_back(std::move(params), std::move(opt));
  }

  // {
  //   auto opt = std::make_unique<torch::optim::AdamOptions>(lr);
  //   opt->betas() = {0.9, 0.99};
  //   opt->eps() = 1e-15;
  //   opt->weight_decay() = 1e-6;

  //   std::vector<Tensor> params;
    // params.push_back(mlp_->params_);
  //   ret.emplace_back(std::move(params), std::move(opt));
  // }

  return ret;
}
std::vector<Tensor> Hash3DAnchored::GetParams()
{
  std::vector<Tensor> ret;
  ret.push_back(feat_pool_);
  // ret.push_back(mlp_->params_);
  return ret;

}
void Hash3DAnchored::Reset() {
  feat_pool_.data().uniform_(-1e-2f, 1e-2f);
  // mlp_->InitParams();
}
void Hash3DAnchored::Zero() {
  feat_pool_.data().zero_();
  // mlp_->InitParams();
}

void Hash3DAnchored::SetFeatPoolRequireGrad(bool require_grad) {
  feat_pool_.requires_grad_(require_grad);
}

void Hash3DAnchored::to(std::string device) {
  torch::Device device_cpu (torch::kCPU);
  torch::Device device_cuda(torch::kCUDA);
  if(device.compare(std::string("cpu")) == 0)
  {
    feat_pool_ = feat_pool_.to(device_cpu);
    
  }
  else
  {

    feat_pool_ = feat_pool_.to(device_cuda);

  }


}



