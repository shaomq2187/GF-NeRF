//
// Created by ppwang on 2022/9/26.
//

#include "PersSampler.h"
// #include "../Utils/Utils.h"
#define MAX_STACK_SIZE 48
#define MAX_OCT_INTERSECT_PER_RAY 1024
#define MAX_SAMPLE_PER_RAY 1024

#define OCC_WEIGHT_BASE 512
#define ABS_WEIGHT_THRES 0.01
#define REL_WEIGHT_THRES 0.1

#define OCC_ALPHA_BASE 32
#define ABS_ALPHA_THRES 0.02
#define REL_ALPHA_THRES 0.1

using Tensor = torch::Tensor;

inline __device__ void GetIntersection(const Wec3f& rays_o,
                                       const Wec3f& rays_d,
                                       const Wec3f& oct_center,
                                       float oct_side_len,
                                       float* near,
                                       float* far) {
  float tmp[3][2];
  float hf_len = oct_side_len * .5f;
#pragma unroll
  for (int64_t i = 0; i < 3; i++) {
    if (rays_d[i] < 1e-6f && rays_d[i] > -1e-6f) {
      if (rays_o[i] > oct_center[i] - hf_len && rays_o[i] < oct_center[i] + hf_len) {
        tmp[i][0] = -1e6f; tmp[i][1] = 1e6f;
      }
      else {
        tmp[i][0] = 1e6f; tmp[i][1] = -1e6f;
      }
    }
    else if (rays_d[i] > 0) {
      tmp[i][0] = (oct_center[i] - hf_len - rays_o[i]) / rays_d[i];
      tmp[i][1] = (oct_center[i] + hf_len - rays_o[i]) / rays_d[i];
    }
    else {
      tmp[i][0] = (oct_center[i] + hf_len - rays_o[i]) / rays_d[i];
      tmp[i][1] = (oct_center[i] - hf_len - rays_o[i]) / rays_d[i];
    }
  }

  near[0] = fmaxf(near[0], fmaxf(tmp[0][0], fmaxf(tmp[1][0], tmp[2][0])));
  far[0] = fminf(far[0], fminf(tmp[0][1], fminf(tmp[1][1], tmp[2][1])));
}

template <bool FILL>
__global__ void FindRayOctreeIntersectionKernel(int64_t n_rays, int64_t max_oct_intersect_per_ray,
                                                uint8_t* search_order,
                                                Wec3f* rays_o_ptr, Wec3f* rays_d_ptr, Wec2f* bounds,
                                                int64_t* oct_idx_counter, Wec2i64* oct_idx_start_end_ptr,
                                                TreeNode* tree_nodes,
                                                int64_t* oct_intersect_idx, Wec2f* oct_intersect_near_far,
                                                int64_t* ) {
  int64_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= n_rays) {
    return;
  }
  // Add offsets
  const Wec3f& rays_o = rays_o_ptr[ray_idx];
  const Wec3f& rays_d = rays_d_ptr[ray_idx];
  Wec2i64& oct_idx_start_end = oct_idx_start_end_ptr[ray_idx];
  // stack_info = stack_info + ray_idx * MAX_STACK_SIZE;
  int64_t stack_info[MAX_STACK_SIZE];
  const float overall_near = bounds[ray_idx][0];
  const float overall_far  = bounds[ray_idx][1];

  int64_t max_intersect_cnt = max_oct_intersect_per_ray;
  if (FILL) {
    max_intersect_cnt = oct_idx_start_end[1] - oct_idx_start_end[0];
    oct_intersect_idx = oct_intersect_idx + oct_idx_start_end[0];
    oct_intersect_near_far = oct_intersect_near_far + oct_idx_start_end[0];
  }

  int64_t stack_ptr = 0;
  int64_t intersect_cnt = 0;

  stack_info[0] = 0;  // Root octree node
  stack_info[1] = -1;

  int64_t ray_st = (int64_t(rays_d[0] > 0.f) << 2) | (int64_t(rays_d[1] > 0.f) << 1) | (int64_t(rays_d[2] > 0.f) << 0);
  search_order += ray_st * 8;
  while (stack_ptr >= 0 && intersect_cnt < max_intersect_cnt) {
    int64_t u = stack_info[stack_ptr * 2]; // Octree node idx;
    const auto& node = tree_nodes[u];
    if (stack_info[stack_ptr * 2 + 1] == -1) {
      float cur_near = overall_near, cur_far = overall_far;
      GetIntersection(rays_o, rays_d, node.center, node.side_len, &cur_near, &cur_far);
      bool can_live_stack = cur_near < cur_far;

      if (can_live_stack) {
        int64_t child_ptr = 0;
        while (child_ptr < 8 && node.childs[search_order[child_ptr]] < 0) {
          child_ptr++;
        }
        if (child_ptr < 8) {   // Has childs, push stack
          stack_info[stack_ptr * 2 + 1] = child_ptr;
          stack_ptr++;
          stack_info[stack_ptr * 2] = node.childs[search_order[child_ptr]];
          stack_info[stack_ptr * 2 + 1] = -1;
        }
        else {
          // Leaf node
          if (node.trans_idx >= 0) {
            if (FILL) {
              oct_intersect_idx[intersect_cnt] = u;
              oct_intersect_near_far[intersect_cnt][0] = cur_near;
              oct_intersect_near_far[intersect_cnt][1] = cur_far;
            }
            intersect_cnt++;
          }
          stack_ptr--;
        }
      }
      else {
        stack_ptr--;
      }
    }
    else {
      int64_t child_ptr = stack_info[stack_ptr * 2 + 1] + 1;
      while (child_ptr < 8 && node.childs[search_order[child_ptr]] < 0) {
        child_ptr++;
      }
      if (child_ptr < 8) {
        stack_info[stack_ptr * 2 + 1] = child_ptr;
        stack_ptr++;
        stack_info[stack_ptr * 2] = node.childs[search_order[child_ptr]];
        stack_info[stack_ptr * 2 + 1] = -1;
      }
      else {
        stack_ptr--;
      }
    }
  }

  if (!FILL) {
    // Phase 1
    int64_t idx_start = atomicAdd((unsigned long long *)oct_idx_counter, (unsigned long)intersect_cnt);
    oct_idx_start_end[0] = idx_start;
    oct_idx_start_end[1] = idx_start + intersect_cnt;
  }
  else {
    // Phase 2
    oct_idx_start_end[1] = oct_idx_start_end[0] + intersect_cnt;
  }
}


void __device__ QueryFrameTransform(const TransInfo& trans,
                                    const Wec3f& cur_xyz,
                                    Wec3f* fill_xyz) {
  Wec4f cur_xyz_ext;
  cur_xyz_ext = cur_xyz.homogeneous();
  Eigen::Matrix<float, N_PROS, 1> transed_vals;
#pragma unroll
  for (int64_t i = 0; i < N_PROS; i++) {
    Wec2f xz = trans.w2xz[i] * cur_xyz_ext;
    transed_vals(i, 0) = xz[0] / xz[1]; 
  }

  Wec3f weighted = trans.weight * transed_vals;
  // weighted = (cur_xyz - trans.center) / (trans.side_len * 0.5) ; // (-1,1)
  *fill_xyz = weighted;
}

void __device__ QueryFrameTransformJac(const TransInfo& trans,
                                       const Wec3f& cur_xyz,
                                       Watrix33f* jac) {
  Wec4f cur_xyz_ext = cur_xyz.homogeneous();
  Eigen::Matrix<float, N_PROS, 3, Eigen::RowMajor> transed_jac;

#pragma unroll
  for (int64_t i = 0; i < N_PROS; i++) {
    Wec2f xz = trans.w2xz[i] * cur_xyz_ext;
    Eigen::Matrix<float, 1, 2, Eigen::RowMajor> dv_dxz;
    dv_dxz(0, 0) = 1 / xz[1]; dv_dxz(0, 1) =-xz[0] / (xz[1] * xz[1]);
    transed_jac.block<1, 3>(i, 0) = dv_dxz * trans.w2xz[i].block<2, 3>(0, 0);
  }

  Watrix33f weighted_jac = trans.weight * transed_jac;
  *jac = weighted_jac;
}

template<bool FILL>
__global__ void RayMarchKernel(int64_t n_rays, float sample_l, bool scale_by_dis,
                               Wec3f* rays_o_ptr, Wec3f* rays_d_ptr, float* rays_noise,
                               Wec2i64* oct_idx_start_end_ptr, int64_t* oct_intersect_idx, Wec2f* oct_intersect_near_far,
                               TreeNode* tree_nodes, TransInfo* transes,
                               Wec2i64* pts_idx_start_end_ptr,
                               Wec3f* sampled_world_pts, Wec3f* sampled_pts, Wec3f* sampled_dirs, Wec3i64* sampled_anchors,
                               float* sampled_dists, float* sampled_ts, int64_t* sampled_oct_idx,
                               float* first_oct_dis) {
  int64_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= n_rays) {
    return;
  }

  rays_noise = rays_noise + ray_idx;
  const auto& rays_o = rays_o_ptr[ray_idx];
  const auto& rays_d = rays_d_ptr[ray_idx];
  const auto& oct_idx_start_end = oct_idx_start_end_ptr[ray_idx];
  oct_intersect_idx = oct_intersect_idx + oct_idx_start_end[0];
  oct_intersect_near_far = oct_intersect_near_far + oct_idx_start_end[0];
  auto& pts_idx_start_end = pts_idx_start_end_ptr[ray_idx];

  int64_t pts_idx = 0;
  if (FILL) {
    int64_t idx_end = pts_idx_start_end[0];
    int64_t idx_cnt = pts_idx_start_end[1];
    pts_idx_start_end[0] = idx_end - idx_cnt;
    pts_idx_start_end[1] = idx_end;
    // pts_idx = pts_idx_start_end[0]; //orig
    pts_idx = ray_idx * MAX_SAMPLE_PER_RAY; // 输入为(n_rays,1024,3)时的寻找方式

    sampled_world_pts = sampled_world_pts + pts_idx;
    sampled_pts = sampled_pts + pts_idx;
    sampled_dirs = sampled_dirs + pts_idx;
    sampled_anchors = sampled_anchors + pts_idx;
    sampled_dists = sampled_dists + pts_idx;
    sampled_ts = sampled_ts + pts_idx;
    sampled_oct_idx = sampled_oct_idx + pts_idx;

    if (oct_idx_start_end[0] < oct_idx_start_end[1]) {
      first_oct_dis[ray_idx] = oct_intersect_near_far[0][0];
    }
    else {
      first_oct_dis[ray_idx] = 1e9f;
    }
  }
  int64_t max_n_samples = FILL ? pts_idx_start_end[1] - pts_idx_start_end[0] : MAX_SAMPLE_PER_RAY;
  max_n_samples = FILL ? min((int)max_n_samples,MAX_SAMPLE_PER_RAY) : 1024; 
  if (max_n_samples <= 0) {
    return;
  }

  int64_t oct_ptr = 0;
  int64_t pts_ptr = 0;
  int64_t cur_oct_idx = oct_intersect_idx[0];
  float cur_march_step = 0.f;
  float exp_march_step = 0.f;

  int64_t n_oct_nodes = oct_idx_start_end[1] - oct_idx_start_end[0];

  float cur_t = oct_intersect_near_far[0][0];
  float cur_far = oct_intersect_near_far[0][1];
  float cur_near = oct_intersect_near_far[0][0];
  Wec3f cur_xyz = rays_o + rays_d * cur_t;
  Wec3f nex_xyz;

  bool the_first_pts = true;
  while (pts_ptr < max_n_samples && oct_ptr < n_oct_nodes) {
    Wec3f fill_xyz = Wec3f::Zero();

    const auto& cur_node = tree_nodes[cur_oct_idx];

    // Get march step
    Watrix33f jac = Watrix33f::Zero();
    const auto& cur_trans = transes[cur_node.trans_idx];
    // const auto& cur_trans = transes[0]; // debug!!!
    float cur_radius = (rays_o - cur_trans.center).norm() / cur_trans.dis_summary;
    float cur_radius_clip = fmaxf(cur_radius, 1.f);
    QueryFrameTransformJac(cur_trans, cur_xyz, &jac);
    Wec3f proj_xyz = jac * rays_d;
    float exp_march_step_warp = sample_l * rays_noise[pts_ptr];
    exp_march_step = exp_march_step_warp / (proj_xyz.norm() + 1e-6f);
    if (scale_by_dis) {
      exp_march_step *= cur_radius_clip;
    }

    cur_march_step = exp_march_step;
    // Do not consider the first point in sampling, because the first point has no randomness in training.
    if (FILL && !the_first_pts) {
      sampled_world_pts[pts_ptr] = cur_xyz;
      sampled_ts[pts_ptr] = cur_t;
      sampled_oct_idx[pts_ptr] = cur_oct_idx;
      sampled_dirs[pts_ptr] = rays_d;

      QueryFrameTransform(cur_trans, cur_xyz, &fill_xyz); //似乎在这里进行了warping
      sampled_dists[pts_ptr] = exp_march_step * (proj_xyz.norm() + 1e-6f);
      sampled_pts[pts_ptr] = fill_xyz;
      sampled_anchors[pts_ptr][0] = cur_node.trans_idx;
      sampled_anchors[pts_ptr][1] = cur_oct_idx;
      sampled_anchors[pts_ptr][2] = cur_node.block_idx;
    }
    if (!the_first_pts) {
      pts_ptr += 1;
    }

    while (cur_t + cur_march_step > cur_far) {
      oct_ptr++;
      if (oct_ptr >= n_oct_nodes) {
        break;
      }
      cur_oct_idx = oct_intersect_idx[oct_ptr];
      cur_near = oct_intersect_near_far[oct_ptr][0];
      cur_far = oct_intersect_near_far[oct_ptr][1];
      int64_t ex_march_steps = ceilf(fmaxf((cur_near - cur_t) / exp_march_step, 1.f));
      cur_march_step = exp_march_step * float(ex_march_steps);
    }
    cur_t += cur_march_step;
    cur_xyz = rays_o + rays_d * cur_t;
    the_first_pts = false;
  }

  if (FILL) {
    pts_idx_start_end[1] = pts_idx_start_end[0] + pts_ptr;
  }
  else {
    pts_idx_start_end[0] = pts_ptr;
    pts_idx_start_end[1] = pts_ptr;
  }
}


std::vector<Tensor>  PersSampler::GetSamples(const Tensor& rays_o_raw, const Tensor& rays_d_raw, const Tensor& bounds_raw) {
  Tensor rays_o = rays_o_raw.contiguous();
  Tensor rays_d = (rays_d_raw / torch::linalg_norm(rays_d_raw, 2, -1, true)).contiguous();
  
  int64_t n_rays = rays_o.sizes()[0];
  Tensor bounds = torch::stack({ torch::full({n_rays}, global_near_, CUDAFloat),
                               torch::full({n_rays}, 1e8f, CUDAFloat) }, -1).contiguous();
  // First, find octree intersections
  Tensor oct_idx_counter = torch::zeros({1}, CUDAInt64);
  Tensor oct_idx_start_end = torch::zeros({ n_rays, 2 }, CUDAInt64);
  Tensor stack_info = torch::zeros({ n_rays * MAX_STACK_SIZE }, CUDAInt64);

  CK_CONT(rays_o);
  CK_CONT(rays_d);
  CK_CONT(oct_idx_counter);
  CK_CONT(oct_idx_start_end);
  CK_CONT(stack_info);
  CK_CONT(bounds);
  CK_CONT(pers_octree_->tree_nodes_gpu_);
  CK_CONT(pers_octree_->pers_trans_gpu_);

  dim3 block_dim = LIN_BLOCK_DIM(n_rays);
  dim3 grid_dim = LIN_GRID_DIM(n_rays);
  auto s1 = std::chrono::high_resolution_clock::now();
  FindRayOctreeIntersectionKernel<false><<<grid_dim, block_dim>>>(
      n_rays, max_oct_intersect_per_ray_,
      pers_octree_->node_search_order_.data_ptr<uint8_t>(),
      RE_INTER(Wec3f*, rays_o.data_ptr()),
      RE_INTER(Wec3f*, rays_d.data_ptr()),
      RE_INTER(Wec2f*, bounds.data_ptr()),
      oct_idx_counter.data_ptr<int64_t>(), RE_INTER(Wec2i64*, oct_idx_start_end.data_ptr()),
      RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
      nullptr, nullptr,
      stack_info.data_ptr<int64_t>());
  auto e1 = std::chrono::high_resolution_clock::now();
  auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(e1 - s1);

  int64_t n_all_oct_intersect = oct_idx_counter.item<int64_t>();
  Tensor oct_intersect_idx = torch::empty({ n_all_oct_intersect }, CUDAInt64);
  Tensor oct_intersect_near_far = torch::empty({ n_all_oct_intersect, 2 }, CUDAFloat);

  FindRayOctreeIntersectionKernel<true><<<grid_dim, block_dim>>>(
      n_rays, max_oct_intersect_per_ray_,
      pers_octree_->node_search_order_.data_ptr<uint8_t>(),
      RE_INTER(Wec3f*, rays_o.data_ptr()),
      RE_INTER(Wec3f*, rays_d.data_ptr()),
      RE_INTER(Wec2f*, bounds.data_ptr()),
      oct_idx_counter.data_ptr<int64_t>(), RE_INTER(Wec2i64*, oct_idx_start_end.data_ptr()),
      RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
      oct_intersect_idx.data_ptr<int64_t>(), RE_INTER(Wec2f*, oct_intersect_near_far.data_ptr()),
      stack_info.data_ptr<int64_t>());

  auto e2 = std::chrono::high_resolution_clock::now();
  auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(e2 - e1);
  // std::cout<<"[debug] perssampler get_samples FindRayOctreeIntersectionKernel[True] time:" << t2.count()<<std::endl;

  // Second, do ray marching
  Tensor pts_idx_start_end = torch::zeros({ n_rays, 2 }, CUDAInt64);

  Tensor rays_noise;
  if (mode_ == RunningMode::VALIDATE) {
    rays_noise = torch::ones({ MAX_SAMPLE_PER_RAY + n_rays + 10 }, CUDAFloat);
  }
  else {
    rays_noise = ((torch::rand({ MAX_SAMPLE_PER_RAY + n_rays + 10 }, CUDAFloat) - .5f) + 1.f).contiguous();
    float sampled_oct_per_ray = float(n_all_oct_intersect) / float(n_rays);
    sampled_oct_per_ray_ = sampled_oct_per_ray_ * .9f + sampled_oct_per_ray * .1f;
  }
  rays_noise.mul_(ray_march_fineness_);
  auto s2 = std::chrono::high_resolution_clock::now();

  RayMarchKernel<false><<<grid_dim, block_dim>>>(
      n_rays, sample_l_, scale_by_dis_,
      RE_INTER(Wec3f*, rays_o.data_ptr()), RE_INTER(Wec3f*, rays_d.data_ptr()),
      rays_noise.data_ptr<float>(),
      RE_INTER(Wec2i64*, oct_idx_start_end.data_ptr()), oct_intersect_idx.data_ptr<int64_t>(), RE_INTER(Wec2f*, oct_intersect_near_far.data_ptr()),
      // unsigned char* occ_bits_tables,
      RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
      RE_INTER(TransInfo*, pers_octree_->pers_trans_gpu_.data_ptr()),
      RE_INTER(Wec2i64*, pts_idx_start_end.data_ptr()),
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
  );
  auto e3 = std::chrono::high_resolution_clock::now();
  auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(e3 - s2);
  // std::cout<<"[debug] perssampler get_samples RayMarchKernel[False] time:" << t3.count()<<std::endl;

  pts_idx_start_end.index_put_({Slc(), 0}, torch::cumsum(pts_idx_start_end.index({Slc(), 0}), 0));

  // int n_all_pts = pts_idx_start_end.index({-1, 0}).item<int>();
  // Tensor sampled_world_pts = torch::empty({ n_all_pts, 3 }, CUDAFloat);
  // Tensor sampled_pts = torch::empty({ n_all_pts, 3 }, CUDAFloat);
  // Tensor sampled_dirs = torch::empty({ n_all_pts, 3 }, CUDAFloat);
  // Tensor sampled_anchors = torch::empty({ n_all_pts, 3 }, CUDAInt);
  // Tensor sampled_dists = torch::empty({ n_all_pts }, CUDAFloat);
  // Tensor sampled_t = torch::empty({ n_all_pts }, CUDAFloat);
  // Tensor sampled_oct_idx = torch::full({ n_all_pts }, -1,CUDAInt).contiguous();
  // Tensor first_oct_dis = torch::zeros({ n_rays, 1 }, CUDAFloat).contiguous();

  // RayMarchKernel<true><<<grid_dim, block_dim>>>(
  //     n_rays, sample_l_, scale_by_dis_,
  //     RE_INTER(Wec3f*, rays_o.data_ptr()), RE_INTER(Wec3f*, rays_d.data_ptr()),
  //     rays_noise.data_ptr<float>(),
  //     RE_INTER(Wec2i*, oct_idx_start_end.data_ptr()), oct_intersect_idx.data_ptr<int>(), RE_INTER(Wec2f*, oct_intersect_near_far.data_ptr()),
  //     // unsigned char* occ_bits_tables,
  //     RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
  //     RE_INTER(TransInfo*, pers_octree_->pers_trans_gpu_.data_ptr()),
  //     RE_INTER(Wec2i*, pts_idx_start_end.data_ptr()),
  //     RE_INTER(Wec3f*, sampled_world_pts.data_ptr()),
  //     RE_INTER(Wec3f*, sampled_pts.data_ptr()),
  //     RE_INTER(Wec3f*, sampled_dirs.data_ptr()),
  //     RE_INTER(Wec3i*, sampled_anchors.data_ptr()),
  //     sampled_dists.data_ptr<float>(), sampled_t.data_ptr<float>(),
  //     sampled_oct_idx.data_ptr<int>(),
  //     first_oct_dis.data_ptr<float>()
  // );
  int64_t n_all_pts = pts_idx_start_end.index({-1, 0}).item<int64_t>();
  Tensor sampled_world_pts = torch::zeros({ n_rays, MAX_SAMPLE_PER_RAY, 3 }, CUDAFloat);
  Tensor sampled_pts = torch::zeros({ n_rays, MAX_SAMPLE_PER_RAY, 3 }, CUDAFloat);
  Tensor sampled_dirs = torch::zeros({ n_rays, MAX_SAMPLE_PER_RAY, 3 }, CUDAFloat);
  Tensor sampled_anchors = torch::zeros({ n_rays, MAX_SAMPLE_PER_RAY, 3 }, CUDAInt64);
  Tensor sampled_dists = torch::zeros({ n_rays, MAX_SAMPLE_PER_RAY }, CUDAFloat);
  Tensor sampled_t = torch::zeros({ n_rays, MAX_SAMPLE_PER_RAY }, CUDAFloat);
  Tensor sampled_oct_idx = torch::full({ n_rays, MAX_SAMPLE_PER_RAY }, -1,CUDAInt64).contiguous();
  Tensor first_oct_dis = torch::zeros({ n_rays, 1 }, CUDAFloat).contiguous();
  auto s3 = std::chrono::high_resolution_clock::now();

  RayMarchKernel<true><<<grid_dim, block_dim>>>(
      n_rays, sample_l_, scale_by_dis_,
      RE_INTER(Wec3f*, rays_o.data_ptr()), RE_INTER(Wec3f*, rays_d.data_ptr()),
      rays_noise.data_ptr<float>(),
      RE_INTER(Wec2i64*, oct_idx_start_end.data_ptr()), oct_intersect_idx.data_ptr<int64_t>(), RE_INTER(Wec2f*, oct_intersect_near_far.data_ptr()),
      // unsigned char* occ_bits_tables,
      RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
      RE_INTER(TransInfo*, pers_octree_->pers_trans_gpu_.data_ptr()),
      RE_INTER(Wec2i64*, pts_idx_start_end.data_ptr()),
      RE_INTER(Wec3f*, sampled_world_pts.data_ptr()),
      RE_INTER(Wec3f*, sampled_pts.data_ptr()),
      RE_INTER(Wec3f*, sampled_dirs.data_ptr()),
      RE_INTER(Wec3i64*, sampled_anchors.data_ptr()),
      sampled_dists.data_ptr<float>(), sampled_t.data_ptr<float>(),
      sampled_oct_idx.data_ptr<int64_t>(),
      first_oct_dis.data_ptr<float>()
  );
  auto e4 = std::chrono::high_resolution_clock::now();
  auto t4 = std::chrono::duration_cast<std::chrono::milliseconds>(e4- s3);
  // std::cout<<"[debug] perssampler get_samples RayMarchKernel[True] time:" << t4.count()<<std::endl;
  return std::vector<Tensor>{
      sampled_world_pts,
      sampled_pts,
      sampled_dirs,
      sampled_dists,
      sampled_t,
      sampled_anchors,
      pts_idx_start_end,
      first_oct_dis,
  };
}

__global__ void GetEdgeSamplesKernel(int64_t n_pts, EdgePool* edge_pool, TransInfo* trans, int64_t* edge_indices, Wec2f* edge_coords,
                                     Wec3f* out_pts, int64_t* out_idx) {
  int64_t pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pts_idx >= n_pts) { return; }
  int64_t edge_idx = edge_indices[pts_idx];
  edge_pool += edge_idx;
  Wec3f world_pts = edge_pool->center + edge_pool->dir_0 * edge_coords[pts_idx][0] + edge_pool->dir_1 * edge_coords[pts_idx][1];
  Wec3f warp_pts_a, warp_pts_b;
  int64_t a = edge_pool->t_idx_a; int64_t b = edge_pool->t_idx_b;
  QueryFrameTransform(trans[a], world_pts, &warp_pts_a);
  QueryFrameTransform(trans[b], world_pts, &warp_pts_b);

  out_pts[pts_idx * 2] = warp_pts_a;
  out_pts[pts_idx * 2 + 1] = warp_pts_b;
  out_idx[pts_idx * 2] = a;
  out_idx[pts_idx * 2 + 1] = b;
}

std::tuple<Tensor, Tensor> PersSampler::GetEdgeSamples(int64_t n_pts) {
  int64_t n_edges = pers_octree_->edge_pool_.size();
  Tensor edge_idx = torch::randint(0, n_edges, { n_pts }, CUDAInt64).contiguous();
  Tensor edge_coord = (torch::rand({n_pts, 2}, CUDAFloat) * 2.f - 1.f).contiguous();
  Tensor out_pts = torch::empty({n_pts, 2, 3}, CUDAFloat).contiguous();
  Tensor out_idx = torch::empty({n_pts, 2}, CUDAInt64).contiguous();

  dim3 block_dim = LIN_BLOCK_DIM(n_pts);
  dim3 grid_dim  = LIN_GRID_DIM(n_pts);

  GetEdgeSamplesKernel<<<grid_dim, block_dim>>>(n_pts,
                                               RE_INTER(EdgePool*, pers_octree_->edge_pool_gpu_.data_ptr()),
                                               RE_INTER(TransInfo*, pers_octree_->pers_trans_gpu_.data_ptr()),
                                               edge_idx.data_ptr<int64_t>(),
                                               RE_INTER(Wec2f*, edge_coord.data_ptr()),
                                               RE_INTER(Wec3f*, out_pts.data_ptr()),
                                               out_idx.data_ptr<int64_t>());

  return { out_pts, out_idx };
}

__global__ void MarkVistNodeKernel(int64_t n_rays,
                                   int64_t* pts_idx_start_end,
                                   int64_t* oct_indices,
                                   float* sampled_weights,
                                   float* sampled_alpha,
                                   int64_t* visit_weight_adder,
                                   int64_t* visit_alpha_adder,
                                                                      int64_t* visit_mark,
                                   int64_t* visit_cnt) {
  const int64_t ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= n_rays) { return; }
  const int64_t pts_idx_start_temp = pts_idx_start_end[ray_idx * 2];
  const int64_t pts_idx_end_temp   = pts_idx_start_end[ray_idx * 2 + 1];

  const int64_t valid_pts_count = pts_idx_end_temp - pts_idx_start_temp; 
  const int64_t pts_idx_start = ray_idx * MAX_SAMPLE_PER_RAY; 
  const int64_t pts_idx_end = pts_idx_start + valid_pts_count;

  if (pts_idx_start >= pts_idx_end) { return; }
  float max_weight = 0.f;
  float max_alpha = 0.f;
  for (int64_t pts_idx = pts_idx_start; pts_idx < pts_idx_end; pts_idx++) {
    max_weight = fmaxf(max_weight, sampled_weights[pts_idx]);
    max_alpha = fmaxf(max_alpha, sampled_alpha[pts_idx]);
  }

  const float weight_thres = fminf(max_weight * REL_WEIGHT_THRES, ABS_WEIGHT_THRES);
  const float alpha_thres = fminf(max_alpha * REL_ALPHA_THRES, ABS_ALPHA_THRES);

  float cur_oct_weight = 0.f;
  float cur_oct_alpha = 0.f;
  int64_t cur_oct_idx = -1;
  int64_t cur_visit_cnt = 0;
  for (int64_t pts_idx = pts_idx_start; pts_idx < pts_idx_end; pts_idx++) {
    if (cur_oct_idx != oct_indices[pts_idx]) {
      if (cur_oct_idx >= 0) {
        atomicMax((long long *)(visit_weight_adder + cur_oct_idx), (long long)(cur_oct_weight > weight_thres ? OCC_WEIGHT_BASE : -1));
        atomicMax((long long *)(visit_alpha_adder + cur_oct_idx), (long long)(cur_oct_alpha > alpha_thres ? OCC_ALPHA_BASE : -1));
        atomicMax((long long *)(visit_cnt + cur_oct_idx), (long long)(cur_visit_cnt));
        visit_mark[cur_oct_idx] = 1;
      }
      cur_oct_idx = oct_indices[pts_idx];
      cur_oct_weight = 0.f;
      cur_oct_alpha = 0.f;
      cur_visit_cnt = 0;
    }
    cur_oct_weight = fmaxf(cur_oct_weight, sampled_weights[pts_idx]);
    cur_oct_alpha = fmaxf(cur_oct_alpha, sampled_alpha[pts_idx]);
    cur_visit_cnt += 1;
  }
  if (cur_oct_idx >= 0) {
    atomicMax((long long *)(visit_weight_adder + cur_oct_idx),(long long ) (cur_oct_weight > weight_thres ? OCC_WEIGHT_BASE : -1));
    atomicMax((long long *)(visit_alpha_adder + cur_oct_idx), (long long ) (cur_oct_alpha > alpha_thres ? OCC_ALPHA_BASE : -1));
    atomicMax((long long *)(visit_cnt + cur_oct_idx), (long long)cur_visit_cnt);
    visit_mark[cur_oct_idx] = 1;
  }
}

__global__ void MarkInvalidNodes(int64_t n_nodes, int64_t* node_weight_stats, int64_t* node_alpha_stats, TreeNode* nodes) {
  int64_t oct_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (oct_idx >= n_nodes) { return; }
  if (node_weight_stats[oct_idx] < 0 || node_alpha_stats[oct_idx] < 0) {
    nodes[oct_idx].trans_idx = -1;
  }
}

void PersSampler::UpdateOctNodes(const Tensor& sampled_anchors,
                                 const Tensor& pts_idx_bounds,
                                 const Tensor& sampled_weight,
                                 const Tensor& sampled_alpha, const int64_t & iter_step) {
  
  // sampled_weight: (n_rays, MAX_SAMPLE_PER_RAY, 1)  
  // sampled_alpha: (n_rays, MAX_SAMPLE_PER_RAY, 1)



  int64_t n_rays = sampled_weight.size(0);
  // Tensor anchors = sample_result[4]; // (n_rays,MAX_SAMPLE_PER_RAY,3)
  // Tensor pts_idx_bounds = sample_result[5]; // (n_rays,MAX_SAMPLE_PER_RAY,2)
   


  // reshape (n_rays, max_sample_per_ray, ...) to (n_rays * max_sample_per_ray, ...):   for compatibility
  Tensor anchors_reshaped = sampled_anchors.reshape({n_rays * MAX_SAMPLE_PER_RAY,3});
  Tensor sampled_weight_reshaped = sampled_weight.reshape({n_rays * MAX_SAMPLE_PER_RAY,1});
  Tensor sampled_alpha_reshaped= sampled_alpha.reshape({n_rays * MAX_SAMPLE_PER_RAY,1});


  const int64_t n_nodes = pers_octree_->tree_nodes_.size();
  const Tensor& pts_idx_start_end = pts_idx_bounds.index({Slc(),0,Slc()}).contiguous();  // (n_rays,2)
  Tensor oct_indices = anchors_reshaped.index({"...", 1}).contiguous();
  CK_CONT(oct_indices);
  CK_CONT(sampled_weight);
  CK_CONT(sampled_alpha);
  CK_CONT(pts_idx_start_end);
  



  Tensor visit_weight_adder = torch::full({ n_nodes }, -1, CUDAInt64);
  Tensor visit_alpha_adder = torch::full({ n_nodes }, -1,CUDAInt64);
  Tensor visit_mark = torch::zeros({ n_nodes }, CUDAInt64);
  Tensor& visit_cnt = pers_octree_->tree_visit_cnt_;
  CK_CONT(visit_weight_adder);
  CK_CONT(visit_alpha_adder);
  CK_CONT(visit_mark);
  CK_CONT(visit_cnt);

  {
    dim3 block_dim = LIN_BLOCK_DIM(n_rays);
    dim3 grid_dim  = LIN_GRID_DIM(n_rays);
    MarkVistNodeKernel<<<grid_dim, block_dim>>>(n_rays,
                                                pts_idx_start_end.data_ptr<int64_t>(),
                                                oct_indices.data_ptr<int64_t>(),
                                                sampled_weight.data_ptr<float>(),
                                                sampled_alpha.data_ptr<float>(),
                                                visit_weight_adder.data_ptr<int64_t>(),
                                                visit_alpha_adder.data_ptr<int64_t>(),
                                                visit_mark.data_ptr<int64_t>(),
                                                visit_cnt.data_ptr<int64_t>());

  }

  Tensor& node_weight_stats = pers_octree_->tree_weight_stats_;
  Tensor occ_weight_mask = (visit_weight_adder > 0).to(torch::kInt32);
  node_weight_stats = torch::maximum(node_weight_stats, occ_weight_mask * visit_weight_adder);
  node_weight_stats += (visit_mark * (1 - occ_weight_mask) * visit_weight_adder);
  node_weight_stats.clamp_(-100, 1 << 20);
  node_weight_stats = node_weight_stats.contiguous();
  CK_CONT(node_weight_stats);

  Tensor& node_alpha_stats = pers_octree_->tree_alpha_stats_;
  Tensor occ_alpha_mask = (visit_alpha_adder > 0).to(torch::kInt32);
  node_alpha_stats = torch::maximum(node_alpha_stats, occ_alpha_mask * visit_alpha_adder);
  node_alpha_stats += (visit_mark * (1 - occ_alpha_mask) * visit_alpha_adder);
  node_alpha_stats.clamp_(-100, 1 << 20);
  node_alpha_stats = node_alpha_stats.contiguous();
  CK_CONT(node_alpha_stats);

  {
    dim3 block_dim = LIN_BLOCK_DIM(n_nodes);
    dim3 grid_dim  = LIN_GRID_DIM(n_nodes);
    MarkInvalidNodes<<<grid_dim, block_dim>>>(
        n_nodes,
        node_weight_stats.data_ptr<int64_t>(),
            node_alpha_stats.data_ptr<int64_t>(),
        RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()));
  }

  while (!sub_div_milestones_.empty() && sub_div_milestones_.back() <= iter_step) {
    pers_octree_->ProcOctree(true, true, sub_div_milestones_.back() <= 0);
    pers_octree_->MarkInvisibleNodes();
    pers_octree_->ProcOctree(true, false, false);
    sub_div_milestones_.pop_back();
  }

  if (iter_step % compact_freq_ == 0) {
    pers_octree_->ProcOctree(true, false, false);
  }
}


__device__ int64_t CheckVisible(const Wec3f& center, float side_len,
                            const Watrix33f& intri, const Watrix34f& w2c, const Wec2f& bound) {
  Wec3f cam_pt = w2c * center.homogeneous();
  float radius = side_len * 0.707;
  if (-cam_pt.z() < bound(0) - radius ||
      -cam_pt.z() > bound(1) + radius) {
    return 0;
  }
  if (cam_pt.norm() < radius) {
    return 1;
  }

  float cx = intri(0, 2);
  float cy = intri(1, 2);
  float fx = intri(0, 0);
  float fy = intri(1, 1);
  float bias_x = radius / -cam_pt.z() * fx;
  float bias_y = radius / -cam_pt.z() * fy;
  float img_pt_x = cam_pt.x() / -cam_pt.z() * fx;
  float img_pt_y = cam_pt.y() / -cam_pt.z() * fy;
  if (img_pt_x + bias_x < -cx || img_pt_x > cx + bias_x ||
      img_pt_y + bias_y < -cy || img_pt_y > cy + bias_y) {
    return 0;
  }
  return 1;
}

__global__ void MarkInvisibleNodesKernel(int64_t n_nodes, int64_t n_cams,
                                         TreeNode* tree_nodes,
                                         Watrix33f* intris, Watrix34f* w2cs, Wec2f* bounds) {
  int64_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_idx >= n_nodes) { return; }
  int64_t n_visible_cams = 0;
  for (int64_t cam_idx = 0; cam_idx < n_cams; cam_idx++) {
    n_visible_cams += CheckVisible(tree_nodes[node_idx].center,
                                   tree_nodes[node_idx].side_len,
                                   intris[cam_idx],
                                   w2cs[cam_idx],
                                   bounds[cam_idx]);
  }
  if (n_visible_cams < 1) {
    tree_nodes[node_idx].trans_idx = -1;
  }
}

void PersOctree::MarkInvisibleNodes() {
  int64_t n_nodes = tree_nodes_.size();
  int64_t n_cams = intri_.size(0);

  CK_CONT(intri_);
  CK_CONT(w2c_);
  CK_CONT(bound_);

  dim3 block_dim = LIN_BLOCK_DIM(n_nodes);
  dim3 grid_dim = LIN_GRID_DIM(n_nodes);
  MarkInvisibleNodesKernel<<<grid_dim, block_dim>>>(
      n_nodes, n_cams,
      RE_INTER(TreeNode*, tree_nodes_gpu_.data_ptr()),
      RE_INTER(Watrix33f*, intri_.data_ptr()),
      RE_INTER(Watrix34f*, w2c_.data_ptr()),
      RE_INTER(Wec2f*, bound_.data_ptr())
  );
}



__global__ void SetBlockIdxsNearestKernel(int64_t n_nodes,int64_t n_block,TreeNode* tree_nodes, Wec3f* centers)
{
  int64_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_idx >= n_nodes) { return; }
  Wec3f node_center = tree_nodes[node_idx].center;


  double min_dist = 1e+9;
  int64_t min_dist_block_idx = -1; 

  for(int64_t block_idx = 0; block_idx < n_block; block_idx++){
    double cur_dis = (node_center - centers[block_idx]).norm();
    if(cur_dis < min_dist){
      min_dist = cur_dis;
      min_dist_block_idx = block_idx;
    }
  }

  tree_nodes[node_idx].block_idx = min_dist_block_idx;

}
void PersOctree::UpdateBlockIdxs(Tensor centers) 
{
  // set block idx according to the nearest distance the blocks' centers
  // center: (n,3)
  CK_CONT(centers);
  CK_CONT(tree_nodes_gpu_);

  std::cout<<"[PersOctree::UpdateBlockIdxs] UpdateBlockIdxs is called" <<std::endl;
  int64_t n_nodes = tree_nodes_.size();
  int64_t n_block = centers.sizes()[0];
  std::cout << "[C++ debug] UpdateBlockIdxs n_nodes: " << n_nodes << std::endl;
  std::cout << "[C++ debug] UpdateBlockIdxs centers: " << centers << std::endl;
  std::cout << "[C++ debug] UpdateBlockIdxs n_block: " << n_block << std::endl;

  dim3 block_dim = LIN_BLOCK_DIM(n_nodes);
  dim3 grid_dim = LIN_GRID_DIM(n_nodes);


  SetBlockIdxsNearestKernel<<<grid_dim, block_dim>>>(
      n_nodes, n_block,
      RE_INTER(TreeNode*, tree_nodes_gpu_.data_ptr()),
      RE_INTER(Wec3f*,centers.data_ptr())
      
  );
  // sync cpu and gpu varibles
  ProcOctree(true,false,false);
  // Tensor tree_nodes_cpu = tree_nodes_gpu_.to(torch::kCPU).contiguous();
  // std::vector<TreeNode> tree_nodes_new;
  // tree_nodes_new.resize(tree_nodes_.size());
  // std::memcpy(RE_INTER(void *, tree_nodes_new.data()), tree_nodes_cpu.data_ptr(), int(tree_nodes_.size() * sizeof(TreeNode)));

}
__global__ void GetRaysTreeNodesIntersectsKernel(int64_t n_rays, int64_t n_nodes,TreeNode* tree_nodes, Wec3f* rays_origins,Wec3f* rays_dirs,float* ts){
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rays * n_nodes) { return; }

  int64_t ray_idx = idx / n_nodes;
  int64_t node_idx = idx % n_nodes;

  Wec3f ray_o = rays_origins[ray_idx];
  Wec3f ray_d = rays_dirs[ray_idx];
  const auto& node = tree_nodes[node_idx];
  float near = -1e6f;
  float far = 1e6f;
  if(!node.is_leaf_node){
    // invalid node: not leaf node
    // regrad trans_idx == -1 as valid node
    ts[idx*2]= far; 
    ts[idx*2 + 1]= near; 
  }
  else{
    GetIntersection(ray_o,ray_d,node.center,node.side_len,&near,&far);
    ts[idx*2]= near; // ts[ray_idx][node_idx][0]  (ray_idx * n_nodes + node_idx) * 2 + 0
    ts[idx*2 + 1]= far; // ts[ray_idx][node_idx][1]  (ray_idx * n_nodes + node_idx) * 2 + 1
  }





  

}

__global__ void GetTreeNodeIdxFromTsKernel(int64_t n_rays, int64_t n_nodes, int64_t n_pts_per_ray, float* t_cur, float* ts, int64_t* anchors){
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rays * n_nodes) { return; }
  int64_t ray_idx = idx / n_nodes;
  int64_t node_idx = idx % n_nodes;


  float node_near = ts[idx*2];
  float node_far = ts[idx*2 + 1];
  if(node_far <= node_near){return;}

  // for each ray and each node, check the intersection of each pt in the ray with the node
  for(int i = 0; i < n_pts_per_ray; i++){
    float cur_pt_t = t_cur[ray_idx * n_pts_per_ray + i]; // t_cur:[2048,128,1]

    if(cur_pt_t >= node_near && cur_pt_t <= node_far){
      anchors[ray_idx * n_pts_per_ray + i] = node_idx; // anchors:[2048,128,1]
    }


  }

}
__global__ void TransQueryFrameKernel(int64_t n_pts,int64_t n_nodes,
       TreeNode* tree_nodes,TransInfo* transes,
       int64_t* anchors,
      Wec3f* world_positions,
      Wec3f* transed_positions){
  // anchors: [2048*128,1]
  // transed_positions: [2048*128,3]
  // world_positions: [2048*128,3]

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pts) { return; }


  int64_t cur_anchor = anchors[idx];
  if(cur_anchor >= n_nodes || cur_anchor < 0 ){return;}

  const auto& cur_node = tree_nodes[cur_anchor];
  if(!cur_node.is_leaf_node){return;}
  

  Wec3f cur_world_pos = world_positions[idx];
  Wec3f cur_transed_pos = Wec3f::Zero();
  if(cur_node.trans_idx >= 0){

    const auto& cur_trans = transes[cur_node.trans_idx];
    QueryFrameTransform(cur_trans,cur_world_pos,&cur_transed_pos);
    transed_positions[idx] = cur_transed_pos;
  }
  else{
    // deal with trans_idx == -1 : no enough visible cameras
    transed_positions[idx] = (cur_world_pos - cur_node.center) / (cur_node.side_len * 0.5);
  }
  



  
  

}
Tensor PersSampler::TransQueryFrame(const Tensor world_positions, const Tensor anchors){
  CK_FLOAT(world_positions);

  CK_CONT(world_positions);
  CK_CONT(anchors);

  int64_t n_pts = world_positions.size(0);
  int64_t n_nodes = pers_octree_->tree_nodes_.size();

  dim3 block_dim = LIN_BLOCK_DIM(n_pts); // thread_per_block
  dim3 grid_dim = LIN_GRID_DIM(n_pts); // block_per_grid
  
  Tensor transed_positions = torch::zeros({n_pts,3},CUDAFloat).contiguous();
  CK_FLOAT(transed_positions);
  CK_CONT(transed_positions);
  TransQueryFrameKernel<<<grid_dim,block_dim>>>(
    n_pts,n_nodes,
    RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
    RE_INTER(TransInfo*, pers_octree_->pers_trans_gpu_.data_ptr()),
    RE_INTER(int64_t*, anchors.data_ptr()),
    RE_INTER(Wec3f*,world_positions.data_ptr()),
    RE_INTER(Wec3f*,transed_positions.data_ptr())
  );
  

  return transed_positions;
  

}

Tensor PersSampler::GetPointsAnchors(const Tensor rays_origins, const Tensor rays_dirs, const Tensor t_starts, const Tensor t_ends){
  // given samples, 
  // rays_origins:(2048,3)
  // rays_dirs:(2048,3)
  // t_starts:(2048,128,1)
  // t_ends: (2048,128,1)
  CK_FLOAT(rays_origins);
  CK_FLOAT(rays_dirs);
  CK_FLOAT(t_starts);
  CK_FLOAT(t_ends);

  CK_CONT(rays_origins);
  CK_CONT(rays_dirs);
  CK_CONT(t_starts);
  CK_CONT(t_ends);

  
  int64_t n_nodes = pers_octree_->tree_nodes_.size();
  int64_t n_rays = rays_origins.size(0);
  int64_t n_pts_per_ray = t_starts.size(1);


  
  Tensor ts = torch::zeros({ n_rays , n_nodes, 2}, CUDAFloat).contiguous();
  Tensor t_cur = (t_starts + t_ends) / 2.0; // (2048,128,1) 
  Tensor anchors = -torch::ones({n_rays, n_pts_per_ray,1},CUDAInt64).contiguous();

  CK_CONT(ts);
  CK_FLOAT(ts);

  CK_CONT(t_cur);
  CK_FLOAT(t_cur);

  CK_CONT(anchors);


  dim3 block_dim = LIN_BLOCK_DIM(n_nodes); // thread_per_block
  dim3 grid_dim = LIN_GRID_DIM(n_nodes * n_rays); // block_per_grid
  GetRaysTreeNodesIntersectsKernel<<<grid_dim,block_dim>>>(
    n_rays,n_nodes,
      RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
      RE_INTER(Wec3f*, rays_origins.data_ptr()),
      RE_INTER(Wec3f*, rays_dirs.data_ptr()),
      RE_INTER(float*, ts.data_ptr())
  );

  GetTreeNodeIdxFromTsKernel<<<grid_dim,block_dim>>>(
    n_rays,n_nodes,n_pts_per_ray,
    RE_INTER(float*,t_cur.data_ptr()),
    RE_INTER(float*,ts.data_ptr()),
    RE_INTER(int64_t*,anchors.data_ptr())
  );


  return anchors;
  
}



__global__ void QueryTreeNodeCenterKernel(int64_t n_pts,int64_t n_nodes,
       TreeNode* tree_nodes,
       int64_t* anchors, Wec3f* centers){ 
  // anchors: [2048*128,1]
  // transed_positions: [2048*128,3]
  // world_positions: [2048*128,3]

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_pts) { return; }


  int64_t cur_anchor = anchors[idx];
  if(cur_anchor >= n_nodes || cur_anchor < 0 ){return;}

  const auto& cur_node = tree_nodes[cur_anchor];
  centers = centers + idx;
  *centers = cur_node.center;
  



}

Tensor PersSampler::QueryTreeNodeCenters(const Tensor& anchors)
{
  // anchors:(n,1)
  int64_t n_pts = anchors.size(0);
  int64_t n_nodes = pers_octree_->tree_nodes_.size();

  CK_CONT(anchors);
  Tensor centers = torch::zeros({n_pts,3},CUDAFloat);
  CK_CONT(centers);
  CK_FLOAT(centers);
  dim3 block_dim = LIN_BLOCK_DIM(n_pts); // thread_per_block
  dim3 grid_dim = LIN_GRID_DIM(n_pts); // block_per_grid
  QueryTreeNodeCenterKernel<<<grid_dim,block_dim>>>(
    n_pts,n_nodes,
    RE_INTER(TreeNode*, pers_octree_->tree_nodes_gpu_.data_ptr()),
    RE_INTER(int64_t*,anchors.data_ptr()),
    RE_INTER(Wec3f*,centers.data_ptr())
  );

  return centers;
  
}

