import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np


def plot_rays(ray_o, ray_d, sample_pts, inside_idxs, file_name):
    # 显示一个批次的光线和tree采样的交点情况
    data = []
    n_rays = ray_o.shape[0]
    n_sample = min(n_rays, 10)
    sample_indices = np.random.choice(n_rays, n_sample, replace=False)
    ray_o = ray_o[sample_indices, :]
    ray_d = ray_d[sample_indices, :]
    sample_pts = sample_pts[sample_indices, :]
    inside_idxs = inside_idxs[sample_indices, :]

    # 画光线
    lines_traces = get_3D_lines_trace(ray_o, ray_o + ray_d * 20)
    data = data + lines_traces

    # 画采样点
    inside_pts_traces = get_inside_points_trace(sample_pts, inside_idxs)
    outside_pts_traces = get_outside_points_trace(sample_pts, inside_idxs)
    data = data + inside_pts_traces
    data = data + outside_pts_traces

    # 输出场景
    fig = go.Figure(data=data)
    scene_dict = dict(
        xaxis=dict(range=[-5, 5], autorange=False),
        yaxis=dict(range=[-5, 5], autorange=False),
        zaxis=dict(range=[-5, 5], autorange=False),
        aspectratio=dict(x=1, y=1, z=1),
    )
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    filename = file_name
    offline.plot(fig, filename=filename, auto_open=False)

def plot_rays_ply(ray_o,ray_d,sample_pts,file_name):
    n_rays = ray_o.shape[0]
    n_sample = min(n_rays, 3)
    sample_indices = np.random.choice(n_rays, n_sample, replace=False)
    ray_o = ray_o[sample_indices, :]
    ray_d = ray_d[sample_indices, :]
    sample_pts = sample_pts[sample_indices, :]

    with open(file_name, "w") as f:
        # write vertices
        for i in range(n_sample):
            start_pt = ray_o[i,:]
            end_pt = (ray_o + ray_d * 20)[i,:]
            f.write(f"v {start_pt[0]} {start_pt[1]} {start_pt[2]}\n")
            f.write(f"v {end_pt[0]} {end_pt[1]} {end_pt[2]}\n")
        
        for i in range(sample_pts.shape[0]):
            for k in range(sample_pts[i,:,:].shape[0]):
                pt = sample_pts[i,k,:]
                f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")

        # write lines
        for i in range(n_sample):
            f.write(f"l {2*i + 1} {2*i + 2}\n")

                
            


def plot_tree_nodes(tree_nodes,file_name,block_idx=0):
    data = []
    centers = tree_nodes.center.cpu().numpy()
    side_lens = tree_nodes.side_len.cpu().numpy()
    n_nodes = centers.shape[0]
    with open(file_name, "w") as f:
        # write vertices
        for i in range(n_nodes):
            center = centers[i].reshape(3,1)
            side_len = side_lens[i]
            for st in range(8):
                offset = np.array([(st >> 2 & 1) - 0.5, (st >> 1 & 1) - 0.5, (st >> 0 & 1) - 0.5]).reshape(3, 1)
                xyz = center + offset * side_len
                f.write(f"v {xyz[0][0]} {xyz[1][0]} {xyz[2][0]}\n")

        for i in range(n_nodes):
            if tree_nodes[i].trans_idx >= 0 and tree_nodes[i].block_idx == block_idx:
                center = centers[i]
                side_len = side_lens[i]
                for a in range(8):
                    for b in range(a + 1, 8):
                        st = a ^ b
                        if st == 1 or st == 2 or st == 4:
                            f.write(f"l {i * 8 + a + 1} {i * 8 + b + 1}\n")


def get_tree_node_trace_line(center,side_len,line_color="blue",line_size = 2,marker_color='grey',marker_size = 1):

    x,y,z = center
    half_length = side_len / 2
        # 正方体的八个顶点坐标
    vertices = [
        (x - half_length, y - half_length, z - half_length),
        (x + half_length, y - half_length, z - half_length),
        (x - half_length, y + half_length, z - half_length),
        (x + half_length, y + half_length, z - half_length),
        (x - half_length, y - half_length, z + half_length),
        (x + half_length, y - half_length, z + half_length),
        (x - half_length, y + half_length, z + half_length),
        (x + half_length, y + half_length, z + half_length)
    ]

    # 定义正方体的边
    edges = [
        (vertices[0], vertices[1]), (vertices[1], vertices[3]),
        (vertices[3], vertices[2]), (vertices[2], vertices[0]),
        (vertices[4], vertices[5]), (vertices[5], vertices[7]),
        (vertices[7], vertices[6]), (vertices[6], vertices[4]),
        (vertices[0], vertices[4]), (vertices[1], vertices[5]),
        (vertices[2], vertices[6]), (vertices[3], vertices[7])
    ]

    # 提取边的坐标
    x_edges = [edge[0][0] for edge in edges]
    y_edges = [edge[0][1] for edge in edges]
    z_edges = [edge[0][2] for edge in edges]

    x_edges += [edge[1][0] for edge in edges]
    y_edges += [edge[1][1] for edge in edges]
    z_edges += [edge[1][2] for edge in edges]

    # 添加边到图形中
    trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color=line_color, width=line_size),
        marker=dict(size=marker_size, color=marker_color)
    )
    return trace
        


def plot_ray_oct_intersections(ray_o, ray_d, sample_pts, inside_idxs, tree_node, file_name):
    # 显示一个批次的光线和tree采样的交点情况
    data = []
    n_rays = ray_o.shape[0]
    n_sample = min(n_rays, 10)
    sample_indices = np.random.choice(n_rays, n_sample, replace=False)
    ray_o = ray_o[sample_indices, :]
    ray_d = ray_d[sample_indices, :]
    sample_pts = sample_pts[sample_indices, :]
    inside_idxs = inside_idxs[sample_indices, :]

    # 画aabb框
    cube_trace = get_tree_node_trace(tree_node)
    data.append(cube_trace)

    # 画光线
    lines_traces = get_3D_lines_trace(ray_o, ray_o + ray_d * 20)
    data = data + lines_traces

    # 画采样点
    inside_pts_traces = get_inside_points_trace(sample_pts, inside_idxs)
    outside_pts_traces = get_outside_points_trace(sample_pts, inside_idxs)
    data = data + inside_pts_traces
    data = data + outside_pts_traces

    # 输出场景
    fig = go.Figure(data=data)
    scene_dict = dict(
        xaxis=dict(range=[-5, 5], autorange=False),
        yaxis=dict(range=[-5, 5], autorange=False),
        zaxis=dict(range=[-5, 5], autorange=False),
        aspectratio=dict(x=1, y=1, z=1),
    )
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    filename = file_name
    offline.plot(fig, filename=filename, auto_open=False)


def get_inside_points_trace(sample_points, inside_idxs):
    traces = []
    n_rays = sample_points.shape[0]
    for i in range(n_rays):
        valid_idx = inside_idxs[i, :]
        points = sample_points[i, valid_idx, :]
        trace = get_3D_scatter_trace(points, name=f"{i}_inside")
        traces.append(trace)
    return traces


def get_outside_points_trace(sample_points, inside_idxs):
    traces = []
    n_rays = sample_points.shape[0]
    for i in range(n_rays):
        valid_idx = ~inside_idxs[i, :]
        points = sample_points[i, valid_idx, :]
        trace = get_3D_scatter_trace(points, name=f"{i}_outside")
        traces.append(trace)
    return traces


def get_tree_node_trace(tree_node):
    x_min = (tree_node.center - tree_node.side_len * 0.5)[0].item()
    y_min = (tree_node.center - tree_node.side_len * 0.5)[1].item()
    z_min = (tree_node.center - tree_node.side_len * 0.5)[2].item()
    x_max = (tree_node.center + tree_node.side_len * 0.5)[0].item()
    y_max = (tree_node.center + tree_node.side_len * 0.5)[1].item()
    z_max = (tree_node.center + tree_node.side_len * 0.5)[2].item()
    i = [1, 1, 2, 3, 3, 0, 0, 1, 4, 4, 0, 1]
    j = [2, 5, 3, 6, 4, 3, 1, 4, 5, 6, 1, 2]
    k = [6, 6, 6, 7, 7, 4, 4, 5, 6, 7, 2, 3]
    x = [x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max]
    y = [y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min]
    z = [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]

    # 创建立方体的三角面
    cube_trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="lightblue", opacity=0.7)

    return cube_trace


def get_3D_lines_trace(start_points, end_points, name="", size=2, caption=None):
    n = start_points.shape[0]
    traces = []
    for i in range(n):
        traces.append(
            go.Scatter3d(
                x=[start_points[i, 0], end_points[i, 0]],
                y=[start_points[i, 1], end_points[i, 1]],
                z=[start_points[i, 2], end_points[i, 2]],
                mode="lines",
                line=dict(width=2),
                name="Line {}".format(i + 1),
            )
        )

    return traces

    # 组合起始点、结束点和线段的数据


def get_3D_scatter_trace(points, name="", size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ),
        text=caption,
    )

    return trace
