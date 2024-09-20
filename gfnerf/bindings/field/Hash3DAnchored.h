//
// Created by ppwang on 2022/7/17.
//

#ifndef SANR_HASH3DANCHORED_H
#define SANR_HASH3DANCHORED_H

#pragma once
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>

#include "TCNNWP.h"
#include "Field.h"

#define N_CHANNELS 2
#define N_LEVELS 16
// 1024
#define RES_FINE_POW_2 10.f
// 8
#define RES_BASE_POW_2 3.f

class Hash3DAnchored : public Field  {
  using Tensor = torch::Tensor;
public:
  Hash3DAnchored(int64_t log2_table_size,int64_t n_volume,double learn_rate);
  std::vector<Tensor> GetParams();

  Tensor AnchoredQuery(const Tensor& points,           // [ n_points, 3 ]
                       const Tensor& anchors           // [ n_points, 3 ]
               ) override;

  int LoadStates(const std::vector<Tensor>& states, int idx) override;
  std::vector<Tensor> States() override;
  std::vector<torch::optim::OptimizerParamGroup> OptimParamGroups() override;
  void Reset() override;
  void Zero() override;
  void SetFeatPoolRequireGrad(bool require_grad) override;
  void to(std::string device)override;
  void ReleaseResources() override;
  int pool_size_;
  // int mlp_hidden_dim_, mlp_out_dim_, n_hidden_layers_;

  Tensor feat_pool_;   // [ pool_size_, n_channels_ ];
  Tensor prim_pool_;   // [ n_levels, 3 ];
  Tensor bias_pool_;   // [ n_levels * n_volumes, 3 ];
  Tensor feat_local_idx_;  // [ n_levels, n_volumes ];
  Tensor feat_local_size_; // [ n_levels, n_volumes ];

  // std::unique_ptr<TCNNWP> mlp_;

  int n_volumes_;
  double learn_rate_;
  bool rand_bias;

  Tensor query_points_, query_volume_idx_;
};

class Hash3DAnchoredInfo : public torch::CustomClassHolder {
public:
  Hash3DAnchored* hash3d_ = nullptr;
  // void createHash3DAnchored(int,int,int,int,int,double);
  
};

namespace torch::autograd {

class Hash3DAnchoredFunction : public Function<Hash3DAnchoredFunction> {
public:
  static variable_list forward(AutogradContext *ctx,
                               Tensor feat_pool_,
                               IValue hash3d_info);

  static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}

#endif //SANR_HASH3DANCHORED_H
