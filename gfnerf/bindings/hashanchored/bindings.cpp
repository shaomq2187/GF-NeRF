#include "../field/Hash3DAnchored.h"
#include "../PtsSampler/PersSampler.h"

// PYBIND11_MODULE(Hash3DAnchored, m) {
// 	// The python bindings expose TCNN's C++ API through
// 	// a single "Module" class that can act as the encoding,
// 	// the neural network, or a combined encoding + network
// 	// under the hood. The bindings don't need to concern
// 	// themselves with these implementation details, though.
//     py::class_<Hash3DAnchored>(m,"Hash3DAnchored")
//     .def(py::init<int, int, int, int,int, float>())
//     .def("AnchoredQuery", &Hash3DAnchored::AnchoredQuery)
//     .def("LoadStates", &Hash3DAnchored::LoadStates)
//     .def("States", &Hash3DAnchored::States)
//     .def("Reset", &Hash3DAnchored::Reset);
		
// }
class m_PersSampler: public torch::CustomClassHolder
{
using Tensor = torch::Tensor;
public:

    m_PersSampler(){}
    void InitSampler( double_t split_dist_thres, std::vector<int64_t> sub_div_milestones,
   int64_t compact_freq, int64_t max_oct_intersect_per_ray,  double_t global_near,
   bool scale_by_dis, int64_t bbox_levels, double_t sample_l,  int64_t max_level, 
   Tensor c2w,  Tensor w2c,  Tensor intri,  Tensor bounds,  int64_t mode,  double_t sampled_oct_per_ray,
   double_t ray_march_fineness, double_t ray_march_init_fineness, int64_t ray_march_fineness_decay_end_iter )
    {
        sampler_ = std::make_unique<PersSampler>(split_dist_thres,sub_div_milestones,
        compact_freq,max_oct_intersect_per_ray,global_near,
        scale_by_dis,bbox_levels,sample_l,max_level,
        c2w,w2c,intri,bounds,mode,sampled_oct_per_ray,
        ray_march_fineness,ray_march_init_fineness,ray_march_fineness_decay_end_iter);
    }
    
    std::vector<Tensor> GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds)
    {
        return sampler_->GetSamples(rays_o,rays_d,bounds);
    }
    Tensor QueryTreeNodeCenters(const Tensor& anchors)
    {
        return sampler_->QueryTreeNodeCenters(anchors);
    }

    // cuda funcs for proposal sampler
    Tensor GetPointsAnchors(const Tensor& rays_origins, const Tensor& rays_dirs, const Tensor& t_starts, const Tensor& t_ends){
        return sampler_->GetPointsAnchors(rays_origins,rays_dirs,t_starts,t_ends);
    }
    Tensor TransQueryFrame(const Tensor& world_positions, const Tensor& anchors){
        return sampler_->TransQueryFrame(world_positions,anchors);
    }
    void VisOctree(std::string base_exp_dir_)
    {
        return sampler_->VisOctree(base_exp_dir_);
    }
    void UpdateOctNodes(const Tensor& sampled_anchors,
                        const Tensor& pts_idx_bounds,
                      const Tensor& sampled_weights,
                      const Tensor& sampled_alpha, const int64_t & iter_step)
    {
        return sampler_->UpdateOctNodes(sampled_anchors,pts_idx_bounds,sampled_weights,sampled_alpha,static_cast<int>(iter_step));
    }
    void UpdateRayMarch(int64_t cur_step)
    {
        return sampler_->UpdateRayMarch(static_cast<int>(cur_step));
    }
    void UpdateMode(int64_t mode)
    {
        return sampler_->UpdateMode(static_cast<int> (mode));
    }
    void UpdateBlockIdxs(Tensor centers)
    {
        return sampler_->pers_octree_->UpdateBlockIdxs(centers);
    }
    std::vector<Tensor> States()
    {
        return sampler_->States();
    }
    int64_t LoadStates(const std::vector<Tensor>& states, int64_t idx)
    {
        return static_cast<int64_t>(sampler_->LoadStates(states,static_cast<int>(idx)));
    }
    std::tuple<Tensor, Tensor> GetEdgeSamples(int64_t n_pts)
    {
        return sampler_->GetEdgeSamples(static_cast<int>(n_pts));
    }
    std::vector<int64_t> get_sub_div_milestones_()
    {
        std::vector<int64_t> results;
        for(auto& value:sampler_->sub_div_milestones_)
        {
            results.push_back(static_cast<int64_t>(value));
        }
        return results;
    }

    // get trans_mat info
    #pragma region
        std::vector<std::vector<std::vector<std::vector<double>>>>  get_pers_trans_w2xz()
        {
            std::vector<std::vector<std::vector<std::vector<double>>>> w2xz;//(n,12,2,4)

            int n_trans = sampler_->pers_octree_->pers_trans_.size();
            for(int i = 0; i < n_trans; i++)
            {
                // w2xz
                std::vector<std::vector<std::vector<double>>> temp;
                for(int k = 0; k < N_PROS; k++)
                {
                    std::vector<std::vector<double>> rows;
                    for(int m=0; m < 2;m++)
                    {
                        std::vector<double> cols;
                        for(int n = 0;n < 4;n++)
                        {
                            cols.push_back(sampler_->pers_octree_->pers_trans_[i].w2xz[k](m,n));
                        }
                        rows.push_back(cols);
                    }
                    temp.push_back(rows);
                }
                w2xz.push_back(temp);
            }
            return w2xz;
        }

        std::tuple<std::vector<std::vector<std::vector<std::vector<double>>>>,
        std::vector<std::vector<std::vector<double>>>
        ,std::vector<std::vector<double>>,
        std::vector<double>,std::vector<double>> get_pers_trans_info()
        {
            std::vector<std::vector<std::vector<std::vector<double>>>> w2xz;//(n,12,2,4)
            std::vector<std::vector<std::vector<double>>> weight;  //(n,3,12)
            std::vector<std::vector<double>> center;
            std::vector<double> side_len;
            std::vector<double> dis_summary;
            
            int n_trans = sampler_->pers_octree_->pers_trans_.size();
            for(int i = 0; i < n_trans; i++)
            {
                // w2xz
                std::vector<std::vector<std::vector<double>>> temp;
                for(int k = 0; k < N_PROS; k++)
                {
                    std::vector<std::vector<double>> rows;
                    for(int m=0; m < 2;m++)
                    {
                        std::vector<double> cols;
                        for(int n = 0;n < 4;n++)
                        {
                            cols.push_back(sampler_->pers_octree_->pers_trans_[i].w2xz[k](m,n));
                        }
                        rows.push_back(cols);
                    }
                    temp.push_back(rows);
                }
                w2xz.push_back(temp);

                // weight
                std::vector<std::vector<double>> rows;
                for(int m=0; m < 3;m++)
                {
                    std::vector<double> cols;
                    for(int n=0; n < 12;n++)
                    {
                        cols.push_back(sampler_->pers_octree_->pers_trans_[i].weight(m,n));
                    }
                    rows.push_back(cols);
                }
                weight.push_back(rows);

                //center
                std::vector<double> cur_center;
                for(int m=0;m<3;m++)
                {
                    cur_center.push_back(sampler_->pers_octree_->pers_trans_[i].center[m]);
                }
                center.push_back(cur_center);
                side_len.push_back(sampler_->pers_octree_->pers_trans_[i].side_len);
                dis_summary.push_back(sampler_->pers_octree_->pers_trans_[i].dis_summary);
            }


            return std::tie(w2xz,weight,center,side_len,dis_summary);
        }

    #pragma endregion
    // get tree_nodes info
    #pragma region
        std::vector<std::vector<double_t>> get_tree_nodes_center_()
        {
            std::vector<std::vector<double_t>> center;

            int n_nodes = sampler_->pers_octree_->tree_nodes_.size();
            for(int i = 0; i < n_nodes;i++)
            {
                std::vector<double_t> temp;
                temp.push_back(static_cast<double_t>(sampler_->pers_octree_->tree_nodes_[i].center[0]));
                temp.push_back(static_cast<double_t>(sampler_->pers_octree_->tree_nodes_[i].center[1]));
                temp.push_back(static_cast<double_t>(sampler_->pers_octree_->tree_nodes_[i].center[2]));

                center.push_back(temp);
            }
            return center;
        }
        std::vector<bool> get_tree_nodes_is_leaf_node_()
        {
            std::vector<bool> is_leaf_node;
            int n_nodes = sampler_->pers_octree_->tree_nodes_.size();
            for(int i = 0; i < n_nodes;i++)
            {
                is_leaf_node.push_back(sampler_->pers_octree_->tree_nodes_[i].is_leaf_node);

            }
            return is_leaf_node;
        }

        std::vector<int64_t> get_tree_nodes_block_idx_()
        {
            std::vector<int64_t> block_idxs;

            int n_nodes = sampler_->pers_octree_->tree_nodes_.size();
            for(int i = 0; i < n_nodes;i++)
            {
                block_idxs.push_back(sampler_->pers_octree_->tree_nodes_[i].block_idx);
            }
            return block_idxs;
        }

        std::vector<double_t> get_tree_nodes_side_len_()
        {
            std::vector<double_t> side_lens;

            int n_nodes = sampler_->pers_octree_->tree_nodes_.size();
            for(int i = 0; i < n_nodes;i++)
            {
                side_lens.push_back(sampler_->pers_octree_->tree_nodes_[i].side_len);
            }
            return side_lens;
        }
        
        std::vector<int64_t> get_tree_nodes_trans_idx_()
        {
            std::vector<int64_t> trans_idxs;

            int n_nodes = sampler_->pers_octree_->tree_nodes_.size();
            for(int i = 0; i < n_nodes;i++)
            {
                trans_idxs.push_back(sampler_->pers_octree_->tree_nodes_[i].trans_idx);
            }
            return trans_idxs;
        }

    # pragma endregion
    int64_t get_compact_freq_()
    {
        return static_cast<int64_t>(sampler_->compact_freq_);
    }
    int64_t get_max_oct_intersect_per_ray_()
    {
        return static_cast<int64_t>(sampler_->max_oct_intersect_per_ray_);
    }
    double_t get_global_near_()
    {
        return static_cast<double_t>(sampler_->global_near_);
    }
    double_t get_sample_l_()
    {
        return static_cast<double_t>(sampler_->sample_l_);
    }
    double_t get_scale_by_dis_()
    {
        return static_cast<double_t>(sampler_->scale_by_dis_);
    }
    int64_t get_mode_()
    {
        return static_cast<int64_t>(sampler_->mode_);
    }
    int64_t get_n_volumes_()
    {
        return static_cast<int64_t>(sampler_->n_volumes_);
    }
    double_t get_sampled_oct_per_ray_()
    {
        return static_cast<double_t>(sampler_->sampled_oct_per_ray_);
    }
    double_t get_ray_march_fineness_()
    {
        return static_cast<double_t>(sampler_->ray_march_fineness_);
    }
    



    std::unique_ptr<PersSampler> sampler_;


};
class m_Hash3DAnchored : public torch::CustomClassHolder
{
using Tensor = torch::Tensor;

public:
    m_Hash3DAnchored(int64_t a, int64_t b, double f)
    {
        hash_3d = std::make_unique<Hash3DAnchored>( a,b, f);
    }
    ~m_Hash3DAnchored(){
        std::cout<<"[m_Hash3DAnchored_CXX] ~m_Hash3DAnchored is called" << std::endl;
        hash_3d.reset();
    }
    void ReleaseResources(){
        hash_3d->ReleaseResources();
    }

    Tensor AnchoredQuery(const Tensor& points,           // [ n_points, 3 ]
                       const Tensor& anchors           // [ n_points, 3 ]
               ){
        return hash_3d->AnchoredQuery(points,anchors);
    }
    int64_t LoadStates(const std::vector<Tensor>& states, int64_t idx) {
        return int64_t(hash_3d->LoadStates(states,int(idx)));
    }
    std::vector<Tensor> States(){
        return hash_3d->States();
    }

    std::vector<Tensor> GetParams() {
        return hash_3d->GetParams();
    };
    void Reset() {hash_3d->Reset();};
    void Zero() {hash_3d->Zero();};

    std::unique_ptr<Field> hash_3d;
    
    void SetFeatPoolRequireGrad(bool require_grad){return hash_3d->SetFeatPoolRequireGrad(require_grad);}
  void to(std::string device){return hash_3d->to(device);}
  
};


TORCH_LIBRARY(my_classes, m) {

    
    m.class_<m_Hash3DAnchored>("Hash3DAnchored")
    .def(torch::init<int64_t, int64_t, double>())
    .def("AnchoredQuery", &m_Hash3DAnchored::AnchoredQuery)
    .def("LoadStates", &m_Hash3DAnchored::LoadStates)
    .def("States", &m_Hash3DAnchored::States)
    .def("Reset", &m_Hash3DAnchored::Reset)
    .def("Zero", &m_Hash3DAnchored::Zero)
    .def("GetParams", &m_Hash3DAnchored::GetParams)
    .def("SetFeatPoolRequireGrad",&m_Hash3DAnchored::SetFeatPoolRequireGrad)
    .def("ReleaseResources",&m_Hash3DAnchored::ReleaseResources)

    .def("to",&m_Hash3DAnchored::to);

    m.class_<m_PersSampler>("PersSampler")
    .def(torch::init<> ())
    .def("InitSampler",&m_PersSampler::InitSampler)
    .def("GetSamples",&m_PersSampler::GetSamples)
    .def("VisOctree",&m_PersSampler::VisOctree)
    .def("UpdateOctNodes",&m_PersSampler::UpdateOctNodes)
    .def("UpdateRayMarch",&m_PersSampler::UpdateRayMarch)
    .def("UpdateMode",&m_PersSampler::UpdateMode)
    .def("UpdateBlockIdxs",&m_PersSampler::UpdateBlockIdxs)
    .def("States",&m_PersSampler::States)
    .def("LoadStates",&m_PersSampler::LoadStates)
    .def("GetEdgeSamples",&m_PersSampler::GetEdgeSamples)
    .def("get_sub_div_milestones_",&m_PersSampler::get_sub_div_milestones_)
    .def("get_compact_freq_",&m_PersSampler::get_compact_freq_)
    .def("get_max_oct_intersect_per_ray_",&m_PersSampler::get_max_oct_intersect_per_ray_)
    .def("get_global_near_",&m_PersSampler::get_global_near_)
    .def("get_sample_l_",&m_PersSampler::get_sample_l_)
    .def("get_scale_by_dis_",&m_PersSampler::get_scale_by_dis_)
    .def("get_mode_",&m_PersSampler::get_mode_)
    .def("get_n_volumes_",&m_PersSampler::get_n_volumes_)
    .def("get_sampled_oct_per_ray_",&m_PersSampler::get_sampled_oct_per_ray_)
    .def("get_ray_march_fineness_",&m_PersSampler::get_ray_march_fineness_)
    
    // tree_nodes info
    .def("get_tree_nodes_center_",&m_PersSampler::get_tree_nodes_center_)
    .def("get_tree_nodes_block_idx_",&m_PersSampler::get_tree_nodes_block_idx_)
    .def("get_tree_nodes_side_len_",&m_PersSampler::get_tree_nodes_side_len_)
    .def("get_tree_nodes_trans_idx_",&m_PersSampler::get_tree_nodes_trans_idx_)
    .def("get_tree_nodes_is_leaf_node_",&m_PersSampler::get_tree_nodes_is_leaf_node_)
    .def("get_pers_trans_info",&m_PersSampler::get_pers_trans_info)
    
    // cuda funcs for proposal sampler
    .def("get_points_anchors",&m_PersSampler::GetPointsAnchors)
    .def("trans_query_frame",&m_PersSampler::TransQueryFrame)


    //cuda funcs for subdataset split
    .def("qurey_tree_nodes_centers",&m_PersSampler::QueryTreeNodeCenters)

    ;

    
}