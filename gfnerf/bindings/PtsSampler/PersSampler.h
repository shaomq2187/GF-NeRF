//
// Created by ppwang on 2022/9/26.
//

#ifndef SANR_PERSSAMPLER_H
#define SANR_PERSSAMPLER_H
#include "Eigen/Eigen"
#include <torch/torch.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include "../Utils/Common.h"
#include "../Utils/Pipe.h"
#define INIT_NODE_STAT 1000
#define N_PROS 12
#define PersMatType Eigen::Matrix<float, 2, 4, Eigen::RowMajor>
#define TransWetType Eigen::Matrix<float, 3, N_PROS, Eigen::RowMajor>

enum RunningMode { TRAIN, VALIDATE };

struct SampleResultFlex {
  using Tensor = torch::Tensor;
  Tensor pts;                           // [ n_all_pts, 3 ]
  Tensor dirs;                          // [ n_all_pts, 3 ]
  Tensor dt;                            // [ n_all_pts, 1 ]
  Tensor t;                             // [ n_all_pts, 1 ]
  Tensor anchors;                       // [ n_all_pts, 3 ]
  Tensor pts_idx_bounds;                // [ n_rays, 2 ] // start, end
  Tensor first_oct_dis;                 // [ n_rays, 1 ]
};
struct alignas(32) TransInfo {
  PersMatType w2xz[N_PROS]; //world to image space
  TransWetType weight; 
  Wec3f center;
  float side_len;
  float dis_summary;
};

struct alignas(32) TreeNode {
  Wec3f center;
  float side_len;
  int64_t parent;
  int64_t childs[8];
  bool is_leaf_node;
  int64_t trans_idx;
  int64_t block_idx;  // for block nerf
  
};

struct alignas(32) EdgePool {
  int64_t t_idx_a;
  int64_t t_idx_b;
  Wec3f center;
  Wec3f dir_0;
  Wec3f dir_1;
};

class PersOctree {
  using Tensor = torch::Tensor;
public:
  PersOctree(int64_t max_depth, float bbox_side_len, float split_dist_thres,
             const Tensor& c2w, const Tensor& w2c, const Tensor& intri, const Tensor& bound);

  std::vector<int64_t> CalcVisiCams(const Tensor& pts);
  void ConstructTreeNode(int64_t u, int64_t depth, Wec3f center, float side_len);
  TransInfo ConstructTrans(const Tensor& rand_pts,
                           const Tensor& c2w,
                           const Tensor& intri,
                           const Tensor& center); // Share intri;
  void ProcOctree(bool compact, bool subdivide, bool brute_force);
  void MarkInvisibleNodes();
  void UpdateBlockIdxs(Tensor centers);  // set block idx according to the nearest distance the blocks' centers

  void ConstructEdgePool();

  int64_t max_depth_;
  Tensor c2w_, w2c_, intri_, bound_;
  float bbox_side_len_;
  float split_dist_thres_;

  std::vector<TreeNode> tree_nodes_;
  Tensor tree_nodes_gpu_;
  Tensor tree_weight_stats_, tree_alpha_stats_;
  Tensor tree_visit_cnt_;
  Tensor node_search_order_;

  std::vector<TransInfo> pers_trans_;
  Tensor pers_trans_gpu_;

  std::vector<EdgePool> edge_pool_;
  Tensor edge_pool_gpu_;
};

class PersSampler:Pipe {
  using Tensor = torch::Tensor;
public:
  PersSampler(const double_t split_dist_thres,const std::vector<int64_t> sub_div_milestones,
  const int64_t compact_freq,const int64_t max_oct_intersect_per_ray, const double_t global_near,
  const bool scale_by_dis,const int64_t bbox_levels,const double_t sample_l, const int64_t max_level, 
  const Tensor c2w, const Tensor w2c, const Tensor intri, const Tensor bounds, const int64_t mode, const double_t sampled_oct_per_ray,
  const double_t ray_march_fineness, const double_t ray_march_init_fineness,const int64_t ray_march_fineness_decay_end_iter
  );

  std::vector<Tensor> GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds);
  Tensor GetPointsAnchors(const Tensor rays_origins, const Tensor rays_directions, const Tensor t_starts, const Tensor t_ends);
  Tensor TransQueryFrame(const Tensor world_positions, const Tensor anchors);

  void VisOctree(std::string base_exp_dir_);
  void VisWarpedPoints(std::string base_exp_dir);
  void UpdateOctNodes(const Tensor& sampled_anchors,
                                  const Tensor& pts_idx_bounds,
                                  const Tensor& sampled_weight,
                                  const Tensor& sampled_alpha, const int64_t & iter_step);
  Tensor QueryTreeNodeCenters(const Tensor& anchors);

  void UpdateMode(int64_t mode);
  void UpdateRayMarch(int64_t cur_step);
  std::vector<Tensor> States() override;
  int LoadStates(const std::vector<Tensor>& states, int idx) override;

  std::tuple<Tensor, Tensor> GetEdgeSamples(int64_t n_pts);

  std::unique_ptr<PersOctree> pers_octree_;

  // global_data_pool
  std::vector<int64_t> sub_div_milestones_;
  int64_t compact_freq_;
  int64_t max_oct_intersect_per_ray_;
  float global_near_;
  float sample_l_;
  bool scale_by_dis_;

  int64_t mode_; // 0: train,  1:eval
  int64_t n_volumes_;
  float sampled_oct_per_ray_;
  float ray_march_fineness_;
  float ray_march_init_fineness_;
  int64_t ray_march_fineness_decay_end_iter_;


};

#endif //SANR_PERSSAMPLER_H
