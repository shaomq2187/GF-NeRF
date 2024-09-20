//
// Created by ppwang on 2022/9/26.
//

#include <random>
#include <algorithm>
#include "../Utils/StopWatch.h"
#include "PersSampler.h"
using Tensor = torch::Tensor;
namespace
{
  float DistanceSummary(const Tensor &dis)
  {
    if (dis.reshape(-1).size(0) <= 0)
    {
      return 1e8f;
    }
    Tensor log_dis = torch::log(dis);
    float thres = torch::quantile(log_dis, 0.25).item<float>();
    Tensor mask = (log_dis < thres).to(torch::kFloat32);
    if (mask.sum().item<float>() < 1e-3f)
    {
      return std::exp(log_dis.mean().item<float>());
    }
    return std::exp(((log_dis * mask).sum() / mask.sum()).item<float>());
  }
  float DistanceSummaryNearest(const Tensor &dis)
  {
    if (dis.reshape(-1).size(0) <= 0 || dis.reshape(-1).size(0) < N_PROS / 2 )
    {
      return 1e8f;
    }
    Tensor log_dis = torch::log(dis);
  
    // float thres = torch::quantile(log_dis, 0.25).item<float>();
    float thres = std::get<0>(torch::kthvalue(log_dis,N_PROS / 2)).item<float>();
    Tensor mask = (log_dis < thres).to(torch::kFloat32);
    if (mask.sum().item<float>() < 1e-3f)
    {
      return std::exp(log_dis.mean().item<float>());
    }
    return std::exp(((log_dis * mask).sum() / mask.sum()).item<float>());
  }

  std::vector<int64_t> GetVisiCams(float bbox_side_len,
                               const Tensor &center,
                               const Tensor &c2w,
                               const Tensor &intri,
                               const Tensor &bound)
  {
    float half_w = intri.index({0, 0, 2}).item<float>();
    float half_h = intri.index({0, 1, 2}).item<float>();
    float cx = intri.index({0, 0, 2}).item<float>();
    float cy = intri.index({0, 1, 2}).item<float>();
    float fx = intri.index({0, 0, 0}).item<float>();
    float fy = intri.index({0, 1, 1}).item<float>();
    int64_t res_w = 128;
    int64_t res_h = std::round(res_w / half_w * half_h);

    Tensor i = torch::linspace(.5f, half_h * 2.f - .5f, res_h, CUDAFloat);
    Tensor j = torch::linspace(.5f, half_w * 2.f - .5f, res_w, CUDAFloat);
    auto ijs = torch::meshgrid({i, j}, "ij");
    i = ijs[0].reshape({-1});
    j = ijs[1].reshape({-1});
    Tensor cam_coords = torch::stack({(j - cx) / fx, -(i - cy) / fy, -torch::ones_like(j, CUDAFloat)}, -1);                                        // [ n_pix, 3 ]
    Tensor rays_d = torch::matmul(c2w.index({Slc(), None, Slc(0, 3), Slc(0, 3)}), cam_coords.index({None, Slc(), Slc(), None})).index({"...", 0}); // [ n_cams, n_pix, 3 ]
    Tensor rays_o = c2w.index({Slc(), None, Slc(0, 3), 3}).repeat({1, res_h * res_w, 1});
    Tensor a = ((center - bbox_side_len * .5f).index({None, None}) - rays_o) / rays_d; 
    Tensor b = ((center + bbox_side_len * .5f).index({None, None}) - rays_o) / rays_d;
    a = torch::nan_to_num(a, 0.f, 1e6f, -1e6f);
    b = torch::nan_to_num(b, 0.f, 1e6f, -1e6f);
    Tensor aa = torch::maximum(a, b); 
    Tensor bb = torch::minimum(a, b);
    auto [far, far_idx] = torch::min(aa, -1);
    auto [near, near_idx] = torch::max(bb, -1);
    far = torch::minimum(far, bound.index({Slc(), None, 1})); 
    near = torch::maximum(near, bound.index({Slc(), None, 0}));
    Tensor mask = (far > near).to(torch::kFloat32).sum(-1); 
    Tensor good = torch::where(mask > 0)[0].to(torch::kInt32).to(torch::kCPU);

    std::vector<int64_t> ret;
    for (int64_t idx = 0; idx < good.sizes()[0]; idx++)
    {
      ret.push_back(good[idx].item<int64_t>());
    }

    return ret;
  }

}

PersOctree::PersOctree(int64_t max_depth, float bbox_side_len, float split_dist_thres,
                       const Tensor &c2w, const Tensor &w2c, const Tensor &intri,
                       const Tensor &bound)
{
  ScopeWatch watch("PersOctree::PersOctree");
  max_depth_ = max_depth;
  bbox_side_len_ = bbox_side_len;
  split_dist_thres_ = split_dist_thres;
  c2w_ = c2w;
  w2c_ = w2c, intri_ = intri;
  bound_ = bound;

  TreeNode root;
  root.parent = -1;
  tree_nodes_.push_back(root);
  ConstructTreeNode(0, 0, Wec3f::Zero(), bbox_side_len);

  // Construct edge pool for edge point sampling - for TV loss.
  ConstructEdgePool();

  // Copy to GPU
  tree_nodes_gpu_ = torch::from_blob(tree_nodes_.data(),
                                     {int64_t(tree_nodes_.size() * sizeof(TreeNode))},
                                     CPUUInt8)
                        .to(torch::kCUDA)
                        .contiguous();

  tree_weight_stats_ = torch::full({int64_t(tree_nodes_.size())}, INIT_NODE_STAT, CUDAInt64);
  tree_alpha_stats_ = torch::full({int64_t(tree_nodes_.size())}, INIT_NODE_STAT, CUDAInt64);
  
  tree_visit_cnt_ = torch::zeros({int64_t(tree_nodes_.size())}, CUDAInt64);
  
  pers_trans_gpu_ = torch::from_blob(pers_trans_.data(),
                                     {int64_t(pers_trans_.size() * sizeof(TransInfo))},
                                     CPUUInt8)
                        .to(torch::kCUDA)
                        .contiguous();
  Tensor pers_trans_tmp = pers_trans_gpu_.to(torch::kCPU).contiguous();

  edge_pool_gpu_ = torch::from_blob(edge_pool_.data(),
                                    {(edge_pool_.size() * sizeof(EdgePool))},
                                    CPUUInt8)
                       .to(torch::kCUDA)
                       .contiguous();

  // Construct octree search order
  std::vector<int64_t> search_order;
  for (int64_t st = 0; st < 8; st++)
  {
    auto cmp = [st](int64_t a, int64_t b)
    {
      int64_t bt = ((a ^ b) & -(a ^ b));
      return (a & bt) ^ (st & bt);
    };

    for (int64_t i = 0; i < 8; i++)
      search_order.push_back(i);
    std::sort(search_order.begin() + st * 8, search_order.begin() + (st + 1) * 8, cmp);
  }
  node_search_order_ = torch::from_blob(search_order.data(), {8 * 8}, CPUInt64).to(torch::kUInt8).to(torch::kCUDA).contiguous();
}

void PersOctree::ProcOctree(bool compact, bool subdivide, bool brute_force)
{
  Tensor tree_nodes_cpu = tree_nodes_gpu_.to(torch::kCPU).contiguous();
  Tensor weight_stats_cpu = tree_weight_stats_.to(torch::kCPU).contiguous();
  int64_t *weight_stats_before = weight_stats_cpu.data_ptr<int64_t>();
  Tensor alpha_stats_cpu = tree_alpha_stats_.to(torch::kCPU).contiguous();
  int64_t *alpha_stats_before = alpha_stats_cpu.data_ptr<int64_t>();
  std::vector<TreeNode> tree_nodes_before;
  tree_nodes_before.resize(tree_nodes_.size());
  std::memcpy(RE_INTER(void *, tree_nodes_before.data()), tree_nodes_cpu.data_ptr(), int64_t(tree_nodes_.size() * sizeof(TreeNode)));

  int64_t n_nodes_before = tree_nodes_before.size();
  PRINT_VAL(n_nodes_before);

  Tensor tree_visit_cnt_cpu = tree_visit_cnt_.to(torch::kCPU).contiguous();
  std::vector<int64_t> visit_cnt(n_nodes_before, 0);
  CHECK_EQ(n_nodes_before, tree_visit_cnt_cpu.size(0));
  std::memcpy(visit_cnt.data(), tree_visit_cnt_cpu.data_ptr<int64_t>(), n_nodes_before * sizeof(int64_t));

  // First, compact tree nodes;
  while (compact)
  {
    for (int64_t u = 0; u < n_nodes_before; u++)
    { 
      if (!tree_nodes_before[u].is_leaf_node)
      {
        
        CHECK_LT(tree_nodes_before[u].trans_idx, 0);
        continue;
      }
      if (tree_nodes_before[u].trans_idx < 0 && tree_nodes_before[u].parent >= 0)
      {
        int64_t v = tree_nodes_before[u].parent;
        for (int64_t st = 0; st < 8; st++)
        {
          if (tree_nodes_before[v].childs[st] == u)
          {
            tree_nodes_before[v].childs[st] = -1;
          }
        }
      }
    }

    bool update_flag = false;
    for (int64_t u = 1; u < n_nodes_before; u++)
    { // root can not be leaf node
      bool has_valid_childs = false;
      for (int64_t st = 0; st < 8; st++)
      {
        if (tree_nodes_before[u].childs[st] >= 0)
        {
          has_valid_childs = true;
          break;
        }
      }
      if (!has_valid_childs)
      {
        if (!tree_nodes_before[u].is_leaf_node)
        {
          update_flag = true;
          CHECK_LT(tree_nodes_before[u].trans_idx, 0);
        }
        tree_nodes_before[u].is_leaf_node = true;
      }
      else
      {
        CHECK(!tree_nodes_before[u].is_leaf_node);
      }
    }

    if (!update_flag)
    {
      break;
    }
  }

  // Compress path
  if (compact)
  {
    auto single_child_func = [&tree_nodes_before](int64_t u)
    {
      int64_t cnt = 0;
      int64_t ret = -1;
      for (int64_t i = 0; i < 8; i++)
      {
        if (tree_nodes_before[u].childs[i] >= 0)
        {
          ret = i;
          cnt++;
        }
      }
      if (cnt == 1)
      {
        return (int)ret;
      }
      return (int)(-1);
    };

    for (int64_t u = 0; u < n_nodes_before; u++)
    {
      if (tree_nodes_before[u].is_leaf_node && tree_nodes_before[u].trans_idx < 0)
      {
        continue;
      }
      int64_t child_idx = -1;
      int64_t v = tree_nodes_before[u].parent;
      int64_t st = -1;
      while (v >= 0 && tree_nodes_before[v].parent >= 0 && (st = single_child_func(v)) >= 0)
      {
        int64_t vv = tree_nodes_before[v].parent;
        for (int64_t i = 0; i < 8; i++)
        {
          if (tree_nodes_before[vv].childs[i] == v)
          {
            tree_nodes_before[vv].childs[i] = u;
          }
        }
        tree_nodes_before[u].parent = vv;
        tree_nodes_before[v].trans_idx = -1;
        tree_nodes_before[v].is_leaf_node = true; // The flag to remove it
        v = vv;
      }
    }
  }

  std::vector<int64_t> new_idx(n_nodes_before, -1);
  std::vector<int64_t> inv_idx;
  int64_t n_nodes_compacted = 0;
  for (int64_t u = 0; u < n_nodes_before; u++)
  {
    if (!tree_nodes_before[u].is_leaf_node || tree_nodes_before[u].trans_idx >= 0)
    {
      new_idx[u] = n_nodes_compacted++;
      inv_idx.push_back(u);
    }
  }
  CHECK_EQ(new_idx[0], 0);
  CHECK_EQ(inv_idx[0], 0);

  std::vector<TreeNode> new_nodes;
  std::vector<int64_t> new_weight_stats;
  std::vector<int64_t> new_alpha_stats;

  for (int64_t u = 0; u < n_nodes_before; u++)
  {
    if (new_idx[u] < 0)
    {
      continue;
    }
    TreeNode node = tree_nodes_before[u];
    if (node.parent >= 0)
    {
      node.parent = new_idx[node.parent];
      CHECK_GE(node.parent, 0);
    }

    for (int64_t st = 0; st < 8; st++)
    {
      if (node.childs[st] >= 0)
      {
        node.childs[st] = new_idx[node.childs[st]];
        CHECK_GE(node.childs[st], 0);
      }
    }

    new_nodes.push_back(node);
    new_weight_stats.push_back(weight_stats_before[u]);
    new_alpha_stats.push_back(alpha_stats_before[u]);
  }

  CHECK_EQ(new_nodes.size(), n_nodes_compacted);
  PRINT_VAL(n_nodes_compacted);

  // Sub-divide


  if (subdivide)
  {
    std::vector<TreeNode> nodes_wp = std::move(new_nodes);
    std::vector<int64_t> weight_stats_wp = std::move(new_weight_stats);
    std::vector<int64_t> alpha_stats_wp = std::move(new_alpha_stats);
    new_nodes.clear();
    new_weight_stats.clear();
    new_alpha_stats.clear();

    std::function<int64_t(int64_t, int64_t)> subdiv_func = [&nodes_wp, &new_nodes,
                                                &weight_stats_wp, &new_weight_stats,
                                                &alpha_stats_wp, &new_alpha_stats,
                                                &visit_cnt, &inv_idx, brute_force,
                                                &subdiv_func](int64_t u, int64_t pa) -> int64_t
    {
      int64_t new_u = new_nodes.size();
      new_nodes.push_back(nodes_wp[u]);
      new_weight_stats.push_back(weight_stats_wp[u]);
      new_alpha_stats.push_back(alpha_stats_wp[u]);
      new_nodes[new_u].parent = pa;

      if (nodes_wp[u].is_leaf_node)
      {
        CHECK(nodes_wp[u].trans_idx >= 0);
        if (!brute_force && visit_cnt[inv_idx[u]] <= 4)
        {
          return new_u;
        }
        for (int64_t st = 0; st < 8; st++)
        {
          Wec3f offset(float((st >> 2) & 1) - .5f, float((st >> 1) & 1) - .5f, float(st & 1) - .5f);
          Wec3f sub_center = new_nodes[new_u].center + new_nodes[new_u].side_len * .5f * offset;

          int64_t v = new_nodes.size();
          new_nodes.emplace_back();
          new_nodes[new_u].childs[st] = v;
          new_nodes[v].center = sub_center;
          new_nodes[v].side_len = new_nodes[new_u].side_len * .5f;
          new_nodes[v].parent = new_u;
          for (int64_t k = 0; k < 8; k++)
            new_nodes[v].childs[k] = -1;
          new_nodes[v].is_leaf_node = true;
          new_nodes[v].trans_idx = new_nodes[new_u].trans_idx;
          // new_nodes[v].block_idx = new_nodes[new_u].block_idx;

          new_weight_stats.push_back(new_weight_stats[new_u]);
          new_alpha_stats.push_back(new_alpha_stats[new_u]);
        }

        new_nodes[new_u].is_leaf_node = false;
        new_nodes[new_u].trans_idx = -1;
        new_weight_stats[new_u] = INIT_NODE_STAT;
        new_alpha_stats[new_u] = INIT_NODE_STAT;
      }
      else
      {
        CHECK(nodes_wp[u].trans_idx < 0);
        for (int64_t st = 0; st < 8; st++)
        {
          if (new_nodes[new_u].childs[st] >= 0)
          {
            int64_t v = subdiv_func(new_nodes[new_u].childs[st], new_u);
            new_nodes[new_u].childs[st] = v;
          }
        }
      }

      return new_u;
    };

    subdiv_func(0, -1);
  }

  CHECK_EQ(new_nodes.size(), new_weight_stats.size());
  CHECK_EQ(new_nodes.size(), new_alpha_stats.size());

  tree_nodes_ = std::move(new_nodes);
  tree_nodes_gpu_ = torch::from_blob(tree_nodes_.data(),
                                     {int64_t(tree_nodes_.size() * sizeof(TreeNode))},
                                     CPUUInt8)
                        .to(torch::kCUDA)
                        .contiguous();

  tree_weight_stats_ = torch::from_blob(new_weight_stats.data(), {int64_t(tree_nodes_.size())}, CPUInt64).to(torch::kCUDA).contiguous();
  tree_alpha_stats_ = torch::from_blob(new_alpha_stats.data(), {int64_t(tree_nodes_.size())}, CPUInt64).to(torch::kCUDA).contiguous();
  tree_visit_cnt_ = torch::zeros({int64_t(tree_nodes_.size())}, CUDAInt64);
  PRINT_VAL(tree_nodes_.size());
}
void PersSampler::VisWarpedPoints(std::string base_exp_dir)
{
  std::cout << "[debug] VisWarpedPoints " << std::endl;
  auto &tree_nodes = pers_octree_->tree_nodes_;
  const int64_t n_rand_pts = 32 * 32 * 32; 
  for (const auto &node : tree_nodes)
  {
    if (node.trans_idx > 0)
    {
      auto &pers_trans = pers_octree_->pers_trans_[node.trans_idx];

      Tensor rand_pts = (torch::rand({n_rand_pts, 3}, CUDAFloat) - .5f) * node.side_len;
      rand_pts.index({Slc(),0}) += node.center[0];
      rand_pts.index({Slc(),1}) += node.center[1];
      rand_pts.index({Slc(),2}) += node.center[2];
      Tensor frame_trans = torch::zeros({N_PROS,2,4},CUDAFloat);
 
      for (int64_t i = 0; i < N_PROS; i++) {
        for (int64_t j = 0; j < 2; j++) {
          for (int64_t k = 0; k < 4; k++) {
            frame_trans[i][j][k] = pers_trans.w2xz[i](j, k);
          }
        }
      }
    Tensor transed_pts = torch::matmul(frame_trans.index({None, Slc(), Slc(), Slc(0, 3)}), rand_pts.index({Slc(), None, Slc(), None}));

    transed_pts = transed_pts.index({"...", 0}) + frame_trans.index({None, Slc(), Slc(), 3});
    transed_pts = transed_pts.index({Slc(), Slc(), 0}) / transed_pts.index({Slc(), Slc(), 1});
    

    transed_pts = transed_pts.permute({1, 0}) ;
    Tensor weight = torch::from_blob(pers_trans.weight.data(), {3, N_PROS}).to(torch::kCUDA); 

    Tensor warped_pts = torch::matmul(weight, transed_pts);
    warped_pts = warped_pts.permute({1,0});
    warped_pts.index({Slc(),0}) += node.center[0];
    warped_pts.index({Slc(),1}) += node.center[1];
    warped_pts.index({Slc(),2}) += node.center[2];
    
    std::ostringstream path;
    path<<base_exp_dir << '/'<<node.trans_idx <<"_transed_pts.obj";

    std::ofstream outFile(path.str()); 
    for (int64_t i = 0; i < warped_pts.size(0); ++i) {
      torch::Tensor pos = warped_pts[i];
      std::ostringstream oss;
      oss << "v " << pos[0].item().toFloat() << " " << pos[1].item().toFloat() << " " << pos[2].item().toFloat() << "\n"; 
      outFile << oss.str(); 
    }
    outFile.close(); 

    }

  }



}


void PersSampler::VisOctree(std::string base_exp_dir_)
{
  // VisWarpedPoints(base_exp_dir_);
  std::ofstream f(base_exp_dir_ + "/octree.obj", std::ios::out);

  auto &tree_nodes = pers_octree_->tree_nodes_;
  int64_t n_nodes = pers_octree_->tree_nodes_.size();
  for (const auto &node : tree_nodes)
  {
    for (int64_t st = 0; st < 8; st++)
    {
      Wec3f xyz = node.center + Wec3f(((st >> 2 & 1) - 0.5f), ((st >> 1 & 1) - 0.5f), ((st >> 0 & 1) - 0.5f)) * node.side_len;
      f << "v " << xyz[0] << " " << xyz[1] << " " << xyz[2] << std::endl;
    }
  }

  for (int64_t i = 0; i < n_nodes; i++)
  {
    if (!tree_nodes[i].is_leaf_node)
    {
      continue;
    }
    for (int64_t a = 0; a < 8; a++)
    {
      for (int64_t b = a + 1; b < 8; b++)
      {
        int64_t st = (a ^ b);
        if (st == 1 || st == 2 || st == 4)
        {
          f << "l " << i * 8 + a + 1 << " " << i * 8 + b + 1 << std::endl;
        }
      }
    }
  }

  f.close();
}

void PersOctree::ConstructTreeNode(int64_t u, int64_t depth, Wec3f center, float side_len)
{
  CHECK_LT(u, tree_nodes_.size());

  tree_nodes_[u].center = center;
  tree_nodes_[u].side_len = side_len;
  tree_nodes_[u].is_leaf_node = false;
  tree_nodes_[u].trans_idx = -1;
  tree_nodes_[u].block_idx = -1;
  for (int64_t i = 0; i < 8; i++)
    tree_nodes_[u].childs[i] = -1;

  if (depth > max_depth_)
  {
    tree_nodes_[u].is_leaf_node = true;
    tree_nodes_[u].trans_idx = -1;
    return;
  }

  int64_t n_all_cams = c2w_.sizes()[0];
  Tensor center_ts = torch::zeros({3}, CPUFloat);
  std::memcpy(center_ts.data_ptr(), &center, 3 * sizeof(float));
  center_ts = center_ts.to(torch::kCUDA);
  const int64_t n_rand_pts = 32 * 32 * 32; 
  Tensor rand_pts = (torch::rand({n_rand_pts, 3}, CUDAFloat) - .5f) * side_len + center_ts.unsqueeze(0);
  auto visi_cams = GetVisiCams(side_len, center_ts, c2w_, intri_, bound_);

  Tensor cam_pos_ts = c2w_.index({Slc(), Slc(0, 3), 3}).to(torch::kCUDA).contiguous();
  Tensor cam_dis = torch::linalg_norm(cam_pos_ts - center_ts.unsqueeze(0), 2, -1, true);
  cam_dis = cam_dis.to(torch::kCPU).contiguous();
  std::vector<float> visi_dis;
  for (int64_t visi_cam : visi_cams)
  {
    float cur_dis = cam_dis[visi_cam].item<float>();
    visi_dis.push_back(cur_dis);
  }
  Tensor visi_dis_ts = torch::from_blob(visi_dis.data(), {int64_t(visi_dis.size())}, CPUFloat).to(torch::kCUDA);
  float distance_summary = DistanceSummary(visi_dis_ts);

  bool exist_unaddressed_cams = (visi_cams.size() >= N_PROS / 2) && (distance_summary < side_len * split_dist_thres_);


  // Subdivide the tree node
  if (exist_unaddressed_cams)
  {
    for (int64_t st = 0; st < 8; st++)
    {
      int64_t v = tree_nodes_.size();
      tree_nodes_.emplace_back();
      Wec3f offset(float((st >> 2) & 1) - .5f, float((st >> 1) & 1) - .5f, float(st & 1) - .5f);
      Wec3f sub_center = center + side_len * .5f * offset;
      tree_nodes_[u].childs[st] = v;
      tree_nodes_[v].parent = u;

      ConstructTreeNode(v, depth + 1, sub_center, side_len * .5f);
    }
  }
  else if (visi_cams.size() < N_PROS / 2)
  {
    tree_nodes_[u].is_leaf_node = true;
    tree_nodes_[u].trans_idx = -1; // Is leaf node but not valid - not enough visible cameras.
  }
  else
  {
    tree_nodes_[u].is_leaf_node = true;
    tree_nodes_[u].trans_idx = pers_trans_.size();
    Tensor visi_cam_c2w = torch::zeros({int64_t(visi_cams.size()), 3, 4}, CUDAFloat);
    for (int64_t i = 0; i < visi_cams.size(); i++)
    {
      visi_cam_c2w.index_put_({i}, c2w_.index({visi_cams[i]}));
    }
    TransInfo ret = ConstructTrans(rand_pts, visi_cam_c2w, intri_[0], center_ts);
    ret.side_len = side_len;
    pers_trans_.push_back(ret);
  }
}

std::tuple<Tensor, Tensor> PCA(const Tensor &pts)
{
  Tensor mean = pts.mean(0, true);

  Tensor moved = pts - mean;
  Tensor cov = torch::matmul(moved.unsqueeze(-1), moved.unsqueeze(1)); // [ n_pts, n_frames, n_frames ];
  cov = cov.mean(0);
 

  auto [L, V] = torch::linalg_eigh(cov);
  L = L.to(torch::kFloat32);
  V = V.to(torch::kFloat32);
  auto [L_sorted, indices] = torch::sort(L, 0, true);
  V = V.permute({1, 0}).contiguous().index({indices}).permute({1, 0}).contiguous(); // { in_dims, 3 }
  L = L.index({indices}).contiguous();


  return {L, V};
}

TransInfo PersOctree::ConstructTrans(const Tensor &rand_pts,
                                     const Tensor &c2w,
                                     const Tensor &intri,
                                     const Tensor &center)
{
  int64_t n_virt_cams = N_PROS / 2;
  int64_t n_cur_cams = c2w.size(0);
  int64_t n_pts = rand_pts.size(0);

  Tensor cam_pos = c2w.index({Slc(), Slc(0, 3), 3}).contiguous();
  Tensor cam_axes = torch::linalg_inv(c2w.index({Slc(), Slc(0, 3), Slc(0, 3)})).contiguous();

  // First step: align distance, find good cameras
  Tensor dis = torch::linalg_norm(cam_pos - center.unsqueeze(0), 2, -1, false);
  float dis_summary = DistanceSummary(dis);

  Tensor rel_cam_pos, normed_cam_pos;

  rel_cam_pos = (cam_pos - center.unsqueeze(0)) / dis.unsqueeze(-1) * dis_summary;
  normed_cam_pos = (cam_pos - center.unsqueeze(0)) / dis.unsqueeze(-1);

  Tensor dis_pairs = torch::linalg_norm(normed_cam_pos.unsqueeze(0) - normed_cam_pos.unsqueeze(1), 2, -1, false);
  dis_pairs = dis_pairs.to(torch::kCPU).contiguous();
  const float *dis_pairs_ptr = dis_pairs.data_ptr<float>();

  std::vector<int64_t> good_cams;
  std::vector<int64_t> cam_marks(n_cur_cams);
  CHECK_GT(n_cur_cams, 0);
  good_cams.push_back(torch::randint(n_cur_cams, {1}, CPUInt64).item<int64_t>());
  cam_marks[good_cams[0]] = 1;

  for (int64_t cnt = 1; cnt < n_virt_cams && cnt < n_cur_cams; cnt++)
  {
    int64_t candi = -1;
    float max_dis = -1.f;
    for (int64_t i = 0; i < n_cur_cams; i++)
    {
      if (cam_marks[i])
        continue;
      float cur_dis = 1e8f;
      for (int64_t j = 0; j < n_cur_cams; j++)
      {
        if (cam_marks[j])
          cur_dis = std::min(cur_dis, dis_pairs_ptr[i * n_cur_cams + j]);
      }
      if (cur_dis > max_dis)
      {
        max_dis = cur_dis;
        candi = i;
      }
    }
    CHECK_GE(candi, 0);
    cam_marks[candi] = 1;
    good_cams.push_back(candi);
  }

  // In case where there are not enough cameras
  for (int64_t i = 0; good_cams.size() < n_virt_cams; i++)
  {
    good_cams.push_back(good_cams[i]);
  }

  // Second step: Construct pers trans
  // At GPU
  Tensor good_cam_scale = torch::ones({n_virt_cams}, CUDAFloat);
  Tensor good_cam_pos = torch::zeros({n_virt_cams, 3}, CUDAFloat);
  Tensor good_rel_cam_pos = torch::zeros({n_virt_cams, 3}, CUDAFloat);
  Tensor good_cam_axis = torch::zeros({n_virt_cams, 3, 3}, CUDAFloat);

  CHECK_EQ(good_cams.size(), n_virt_cams);

  Tensor cam_scale = (dis / dis_summary).clip(1.f, 1e9f);
  rel_cam_pos = (cam_pos - center.unsqueeze(0)) / dis.unsqueeze(-1) * dis.unsqueeze(-1).clip(dis_summary, 1e9f);

  for (int64_t i = 0; i < good_cams.size(); i++)
  {
    good_cam_pos.index_put_({i}, (rel_cam_pos[good_cams[i]] + center));
    good_rel_cam_pos.index_put_({i}, (rel_cam_pos[good_cams[i]]));
    good_cam_axis.index_put_({i}, cam_axes[good_cams[i]]);
    good_cam_scale.index_put_({i}, cam_scale[good_cams[i]]);
  }

  Tensor expect_z_axis = good_rel_cam_pos / torch::linalg_norm(good_rel_cam_pos, 2, -1, true);
  Tensor rots = torch::zeros({n_virt_cams, 3, 3}, CUDAFloat);

  auto ToEigenVec3 = [](Tensor x)
  {
    Wec3f ret;
    x = x.to(torch::kCPU);
    for (int64_t i = 0; i < 3; i++)
    {
      ret(i) = x[i].item<float>();
    }
    return ret;
  };

  auto ToTorchMat33 = [](Watrix33f x)
  {
    Tensor ret = torch::zeros({3, 3}, CPUFloat);
    for (int64_t i = 0; i < 3; i++)
    {
      for (int64_t j = 0; j < 3; j++)
      {
        ret.index_put_({i, j}, x(i, j));
      }
    }
    return ret.to(torch::kCUDA);
  };

  for (int64_t i = 0; i < good_cams.size(); i++)
  {
    Wec3f from_z_axis = ToEigenVec3(good_cam_axis.index({i, 2, Slc(0, 3)}));
    Wec3f to_z_axis = ToEigenVec3(expect_z_axis.index({i, Slc(0, 3)}));
    Wec3f crossed = from_z_axis.cross(to_z_axis);
    float cos_val = from_z_axis.dot(to_z_axis);
    float sin_val = crossed.norm();

    sin_val = std::max(-0.999999f,std::min(sin_val,0.999999f)); // clamp
    cos_val = std::max(-0.999999f,std::min(cos_val,0.999999f)); // clamp
    
    float angle = std::asin(sin_val);
    if (cos_val < 0.f)
    {
      angle = M_PI - angle;
    }
    crossed = crossed.normalized();
    Watrix33f rot_mat;

    rot_mat = Eigen::AngleAxisf(angle, crossed);

    rots.index_put_({i}, ToTorchMat33(rot_mat));
  }

  good_cam_axis = torch::matmul(good_cam_axis, rots.transpose(1, 2));

  Tensor x_axis = good_cam_axis.index({Slc(), 0, Slc()}).contiguous();
  Tensor y_axis = good_cam_axis.index({Slc(), 1, Slc()}).contiguous();
  Tensor z_axis = good_cam_axis.index({Slc(), 2, Slc()}).contiguous();
  Tensor diff = z_axis - expect_z_axis;

  // CHECK_LT(diff.abs().max().item<float>(), 2e-3f);

  float focal = (intri.index({0, 0}) / intri.index({0, 2})).item<float>();
  x_axis *= focal;
  y_axis *= focal;
  x_axis *= good_cam_scale.unsqueeze(-1);
  y_axis *= good_cam_scale.unsqueeze(-1);
  x_axis = torch::cat({x_axis, y_axis}, 0);
  z_axis = torch::cat({z_axis, z_axis}, 0);

  Tensor wp_cam_pos = torch::cat({good_cam_pos, good_cam_pos}, 0);
  Tensor frame_trans = torch::zeros({N_PROS, 2, 4}, CUDAFloat);
  frame_trans.index_put_({Slc(), 0, Slc(0, 3)}, x_axis);
  frame_trans.index_put_({Slc(), 1, Slc(0, 3)}, z_axis);
  frame_trans.index_put_({Slc(), 0, 3}, -(x_axis * wp_cam_pos).sum(-1));
  frame_trans.index_put_({Slc(), 1, 3}, -(z_axis * wp_cam_pos).sum(-1));

  // Third step: Construct frame weight by PCA.
  // Mapped points and Jacobian
  // std::cout << "[debug] frame_trans shape:" << frame_trans.index({None, Slc(), Slc(), Slc(0, 3)}).sizes() << std::endl;
  // std::cout << "[debug] rand_pts shape:" << rand_pts.index({Slc(), None, Slc(), None}).sizes() << std::endl;

  Tensor transed_pts = torch::matmul(frame_trans.index({None, Slc(), Slc(), Slc(0, 3)}), rand_pts.index({Slc(), None, Slc(), None}));
  transed_pts = transed_pts.index({"...", 0}) + frame_trans.index({None, Slc(), Slc(), 3});
  
  // std::cout << "[debug] transed_pts shape:" << transed_pts.index({"...", 0}).sizes() << std::endl;
  // std::cout << "[debug] frame_trans shape:" << frame_trans.index({None, Slc(), Slc(), 3}).sizes()<< std::endl;

  Tensor dv_da = 1.f / transed_pts.index({Slc(), Slc(), 1});
  Tensor dv_db = transed_pts.index({Slc(), Slc(), 0}) / -transed_pts.index({Slc(), Slc(), 1}).square();
  Tensor dv_dab = torch::stack({dv_da, dv_db}, -1);                                              // [ n_pts, N_PROS, 2 ]
  Tensor dab_dxyz = frame_trans.index({None, Slc(), Slc(), Slc(0, 3)}).clone();                  // [ n_pts, N_PROS, 2, 3 ];
  Tensor dv_dxyz = torch::matmul(dv_dab.unsqueeze(2), dab_dxyz).index({Slc(), Slc(), 0, Slc()}); // [ n_pts, N_PROS, 3 ];
  
  // std::cout << "[debug] dab_dxyz shape :" << dab_dxyz.sizes()<< std::endl;
  // std::cout << "[debug] transed_pts shape 0 :" << transed_pts.sizes()<< std::endl;

  CHECK(transed_pts.index({Slc(), Slc(), 1}).max().item<float>() < 0.f);
  // std::cout << "[debug] transed_pts shape 1 :" << transed_pts.index({Slc(), Slc(), 1}).sizes()<< std::endl;
  // std::cout << "[debug] transed_pts  0 :" << transed_pts.index({Slc(), Slc(), 0}).sizes()<< std::endl;
  // std::cout << "[debug] transed_pts  1 :" << transed_pts.index({Slc(), Slc(), 1}).sizes()<< std::endl;

  transed_pts = transed_pts.index({Slc(), Slc(), 0}) / transed_pts.index({Slc(), Slc(), 1});
  // std::cout << "[debug] transed_pts 2 min:" << transed_pts.min() << std::endl;
  // std::cout << "[debug] transed_pts 2 max:" << transed_pts.max() << std::endl;

  CHECK_NOT_NAN(transed_pts);

  // Cosntruct lin mapping
  Tensor L, V;
  std::tie(L, V) = PCA(transed_pts);
  V = V.permute({1, 0}).index({Slc(0, 3)}).contiguous(); // [ 3, N_PROS ]
  // std::cout << "[debug] V  shape:" << V.index({None}).sizes()<< std::endl;

  Tensor jac = torch::matmul(V.index({None}), dv_dxyz); // [ n_pts, 3, 3 ];
  Tensor jac_warp2world = torch::linalg_inv(jac);
  Tensor jac_warp2image = torch::matmul(dv_dxyz, jac_warp2world);

  Tensor jac_abs = jac_warp2image.abs();            // [n_pts, N_PROS, 3]
  auto [jac_max, max_tmp] = torch::max(jac_abs, 1); // [ n_pts, 3 ]
  Tensor exp_step = 1.f / jac_max;                  // [n_pts, 3];
  Tensor mean_step = exp_step.mean(0);
  V /= mean_step.unsqueeze(-1);

  Tensor V_cpu = V.to(torch::kCPU).contiguous();
  Tensor frame_trans_cpu = frame_trans.to(torch::kCPU).contiguous();

  CHECK_NOT_NAN(V_cpu);
  CHECK_NOT_NAN(frame_trans_cpu);
  TransInfo ret;
  std::memcpy(&(ret.w2xz), frame_trans_cpu.data_ptr(), sizeof(PersMatType) * N_PROS);
  std::memcpy(ret.weight.data(), V_cpu.data_ptr(), sizeof(TransWetType));
  for (int64_t i = 0; i < 3; i++)
  {
    ret.center[i] = center[i].item<float>();
  }
  ret.dis_summary = dis_summary;
  return ret;
}

void PersOctree::ConstructEdgePool()
{
  ScopeWatch watch("PersOctree::ConstructEdgePool");
  int64_t n_nodes = tree_nodes_.size();

  auto is_inside = [](const TreeNode &node, const Wec3f &pt) -> bool
  {
    Wec3f bias = (pt - node.center) / node.side_len * 2.f;
    return bias.cwiseAbs().maxCoeff() < (1.f + 1e-4f);
  };

  for (int64_t a = 0; a < n_nodes; a++)
  {
    if (tree_nodes_[a].trans_idx < 0)
    {
      continue;
    }
    for (int64_t b = a + 1; b < n_nodes; b++)
    {
      if (tree_nodes_[b].trans_idx < 0)
      {
        continue;
      }

      int64_t u = a, v = b;
      int64_t t_a = tree_nodes_[a].trans_idx;
      int64_t t_b = tree_nodes_[b].trans_idx;
      if (tree_nodes_[u].side_len > tree_nodes_[v].side_len)
      {
        std::swap(u, v);
      }

      float len_u = tree_nodes_[u].side_len * .5f;
      const Wec3f &ct_u = tree_nodes_[u].center;
      if (is_inside(tree_nodes_[v], ct_u + Wec3f(len_u, 0.f, 0.f)))
      {
        edge_pool_.push_back({t_a, t_b, ct_u + Wec3f(len_u, 0.f, 0.f), Wec3f(0.f, len_u, 0.f), Wec3f(0.f, 0.f, len_u)});
      }
      if (is_inside(tree_nodes_[v], ct_u - Wec3f(len_u, 0.f, 0.f)))
      {
        edge_pool_.push_back({t_a, t_b, ct_u - Wec3f(len_u, 0.f, 0.f), Wec3f(0.f, len_u, 0.f), Wec3f(0.f, 0.f, len_u)});
      }
      if (is_inside(tree_nodes_[v], ct_u + Wec3f(0.f, len_u, 0.f)))
      {
        edge_pool_.push_back({t_a, t_b, ct_u + Wec3f(0.f, len_u, 0.f), Wec3f(len_u, 0.f, 0.f), Wec3f(0.f, 0.f, len_u)});
      }
      if (is_inside(tree_nodes_[v], ct_u - Wec3f(0.f, len_u, 0.f)))
      {
        edge_pool_.push_back({t_a, t_b, ct_u - Wec3f(0.f, len_u, 0.f), Wec3f(len_u, 0.f, 0.f), Wec3f(0.f, 0.f, len_u)});
      }
      if (is_inside(tree_nodes_[v], ct_u + Wec3f(0.f, 0.f, len_u)))
      {
        edge_pool_.push_back({t_a, t_b, ct_u + Wec3f(0.f, 0.f, len_u), Wec3f(len_u, 0.f, 0.f), Wec3f(0.f, len_u, 0.f)});
      }
      if (is_inside(tree_nodes_[v], ct_u - Wec3f(0.f, 0.f, len_u)))
      {
        edge_pool_.push_back({t_a, t_b, ct_u - Wec3f(0.f, 0.f, len_u), Wec3f(len_u, 0.f, 0.f), Wec3f(0.f, len_u, 0.f)});
      }
    }
  }

  PRINT_VAL(edge_pool_.size());
}

// -------------------------------------------- Sampler ------------------------------------------------

PersSampler::PersSampler(const double_t split_dist_thres,const std::vector<int64_t> sub_div_milestones,
const int64_t compact_freq,const int64_t max_oct_intersect_per_ray, const double_t global_near,
const bool scale_by_dis,const int64_t bbox_levels,const double_t sample_l, const int64_t max_level, 
const Tensor c2w, const Tensor w2c, const Tensor intri, const Tensor bounds, const int64_t mode, const double_t sampled_oct_per_ray,
const double_t ray_march_fineness,const double_t ray_march_init_fineness,const int64_t ray_march_fineness_decay_end_iter
)
{
  ScopeWatch watch("PersSampler::PersSampler");
  
  float split_dist_thres_ = static_cast<float>(split_dist_thres);
  for(const auto & value : sub_div_milestones)
  {
    sub_div_milestones_.push_back(static_cast<int32_t>(value));
  }
  compact_freq_ = static_cast<int64_t>(compact_freq);
  max_oct_intersect_per_ray_ = static_cast<int64_t>(max_oct_intersect_per_ray);
  std::reverse(sub_div_milestones_.begin(), sub_div_milestones_.end());

  global_near_ = static_cast<float>(global_near);
  scale_by_dis_ = scale_by_dis;
  int64_t bbox_levels_ =static_cast<int64_t>(bbox_levels);

  float bbox_side_len = (1 << (bbox_levels_ - 1));

  sample_l_ = static_cast<float>(sample_l);
  int64_t max_level_ = static_cast<int64_t>(max_level);
  mode_ = static_cast<int64_t>(mode);
  sampled_oct_per_ray_ = static_cast<float>(sampled_oct_per_ray);
  ray_march_fineness_ = static_cast<float>(ray_march_fineness);
  ray_march_init_fineness_ = static_cast<float>(ray_march_init_fineness);
  ray_march_fineness_decay_end_iter_ = static_cast<float>(ray_march_fineness_decay_end_iter);

  pers_octree_ = std::make_unique<PersOctree>(
      max_level_, bbox_side_len, split_dist_thres_,
      c2w, w2c, intri, bounds);


  n_volumes_ = pers_octree_->pers_trans_.size();
  std::cout<<"[debug] persampler::split_dist_thres: "<<split_dist_thres<<std::endl;
  std::cout<<"[debug] persampler::sub_div_milestones: "<<sub_div_milestones_<<std::endl;
  std::cout<<"[debug] persampler::compact_freq: "<<compact_freq_<<std::endl;
  std::cout<<"[debug] persampler::max_oct_intersect_per_ray: "<<max_oct_intersect_per_ray_<<std::endl;
  std::cout<<"[debug] persampler::global_near: "<<global_near_<<std::endl;
  std::cout<<"[debug] persampler::scale_by_dis: "<<scale_by_dis_<<std::endl;
  std::cout<<"[debug] persampler::bbox_levels: "<<bbox_levels<<std::endl;
  std::cout<<"[debug] persampler::sample_l: "<<sample_l_<<std::endl;
  std::cout<<"[debug] persampler::max_level: "<<max_level<<std::endl;
  std::cout<<"[debug] persampler::ray_march_fineness: "<<ray_march_fineness_<<std::endl;
  std::cout<<"[debug] persampler::ray_march_init_fineness_: "<<ray_march_init_fineness_<<std::endl;
  std::cout<<"[debug] persampler::ray_march_fineness_decay_end_iter_: "<<ray_march_fineness_decay_end_iter_<<std::endl;
  std::cout<<"[debug] persampler::sampled_oct_per_ray_: "<<sampled_oct_per_ray_<<std::endl;


}
void PersSampler::UpdateMode(int64_t mode)
{
  mode_ = mode;
}

void PersSampler::UpdateRayMarch(int64_t cur_step)
{
  if (cur_step >= ray_march_fineness_decay_end_iter_) {
    ray_march_fineness_ = 1.f;
  }
  else {
    float progress = float(cur_step) / float(ray_march_fineness_decay_end_iter_);
    ray_march_fineness_ = std::exp(std::log(1.f) * progress + std::log(ray_march_init_fineness_) * (1.f - progress));
  }
}

std::vector<Tensor> PersSampler::States()
{
  std::vector<Tensor> ret;
  ret.push_back(pers_octree_->tree_nodes_gpu_);
  ret.push_back(pers_octree_->pers_trans_gpu_);
  ret.push_back(pers_octree_->tree_visit_cnt_);
  Tensor milestones_ts = torch::from_blob(sub_div_milestones_.data(), sub_div_milestones_.size(), CPUInt64).to(torch::kCUDA);
  ret.push_back(milestones_ts);

  return ret;
}



int PersSampler::LoadStates(const std::vector<Tensor> &states, int idx)
{
  pers_octree_->tree_nodes_gpu_ = states[idx++].clone().to(torch::kCUDA).contiguous();
  pers_octree_->pers_trans_gpu_ = states[idx++].clone().to(torch::kCUDA).contiguous();
  pers_octree_->tree_visit_cnt_ = states[idx++].clone().to(torch::kCUDA).contiguous();
  Tensor milestones_ts = states[idx++].clone().to(torch::kCPU).contiguous();

  Tensor tree_nodes_cpu = pers_octree_->tree_nodes_gpu_.to(torch::kCPU);
  pers_octree_->tree_nodes_.resize(tree_nodes_cpu.sizes()[0] / sizeof(TreeNode));
  std::memcpy(pers_octree_->tree_nodes_.data(), tree_nodes_cpu.data_ptr(), tree_nodes_cpu.sizes()[0]);

  Tensor pers_trans_cpu = pers_octree_->pers_trans_gpu_.to(torch::kCPU);
  pers_octree_->pers_trans_.resize(pers_trans_cpu.sizes()[0] / sizeof(TransInfo));
  std::memcpy(pers_octree_->pers_trans_.data(), pers_trans_cpu.data_ptr(), pers_trans_cpu.sizes()[0]);

  sub_div_milestones_.resize(milestones_ts.sizes()[0]);
  std::memcpy(sub_div_milestones_.data(), milestones_ts.data_ptr(), milestones_ts.sizes()[0] * sizeof(int64_t));
  PRINT_VAL(sub_div_milestones_);

  pers_octree_->tree_weight_stats_ = torch::full({int64_t(pers_octree_->tree_nodes_.size())}, INIT_NODE_STAT, CUDAInt64);
  pers_octree_->tree_alpha_stats_ = torch::full({int64_t(pers_octree_->tree_nodes_.size())}, INIT_NODE_STAT, CUDAInt64);

  int64_t valid_nodes = 0;
  for (int64_t i = 0; i < pers_octree_->tree_nodes_.size(); i++)
  {
    if (pers_octree_->tree_nodes_[i].trans_idx >= 0)
    {
      valid_nodes++;
    }
  }
  PRINT_VAL(valid_nodes);

  return idx;
}

