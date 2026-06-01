import pymeshfix as mf
import pyvista as pv
import numpy as np
import trimesh
import open3d as o3d
import subprocess
import sys
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# =============================================================================
# 第一步：辅助函数 - 过滤 Mesh 边界（尺寸过滤 + 长宽比过滤）
# =============================================================================

def filter_boundary_holes(components, tube_vertices, mesh_bbox_diagonal):
    """
    过滤掉尺寸过大的"孔洞"，这些很可能是 mesh 边界。
    
    启发式规则：
    - 如果孔洞的包围盒对角线 > mesh 对角线的 20%，认为是边界
    
    Args:
        components: 连通分量列表，每个元素是顶点索引数组
        tube_vertices: 管道顶点的坐标数组
        mesh_bbox_diagonal: mesh 包围盒的对角线长度
    
    Returns:
        filtered_components: 过滤后的孔洞分量列表
    """
    MIN_SIZE_RATIO = 0.01   # 最小尺寸占 mesh 的比例（太小可能是噪声）
    MAX_SIZE_RATIO = 0.25   # 最大尺寸占 mesh 的比例（太大可能是边界）
    MAX_VERTEX_COUNT = 5000  # 最大顶点数（过多可能是多个环相连）
    MIN_VERTEX_COUNT = 5     # 最小顶点数
    
    filtered = []
    for comp in components:
        # 跳过空分量
        if len(comp) == 0:
            continue
        
        pts = tube_vertices[comp]
        
        # 尺寸检查
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        size_ratio = bbox_diag / mesh_bbox_diagonal
        
        # 顶点数检查
        vertex_count = len(np.unique(comp))
        
        # 判断：是否是真实孔洞
        is_valid_size = MIN_SIZE_RATIO < size_ratio < MAX_SIZE_RATIO
        is_valid_count = MIN_VERTEX_COUNT <= vertex_count <= MAX_VERTEX_COUNT
        
        if is_valid_size and is_valid_count:
            filtered.append(comp)
            print(f"  有效孔洞: 尺寸比例={size_ratio:.3f}, 顶点数={vertex_count}")
        else:
            pass
            # print(f"  过滤边界: 尺寸比例={size_ratio:.3f}, 顶点数={vertex_count}, "
            #       f"原因={'太大' if size_ratio >= MAX_SIZE_RATIO else '太小' if size_ratio <= MIN_SIZE_RATIO else ''}"
            #       f"{'过多顶点数' if vertex_count > MAX_VERTEX_COUNT else '过少顶点数' if vertex_count < MIN_VERTEX_COUNT else ''}")
    
    return filtered


# =============================================================================
# 第二步：辅助函数 - 合并相邻孔洞（Union-Find）
# =============================================================================

def merge_nearby_components(components, tube_vertices, merge_distance_ratio=1.5):
    """
    合并距离过近的连通分量，避免选择相邻孔洞导致的重复采样。
    
    思路：如果两个孔洞的中心距离 < max(两者尺寸)，则视为"相邻孔洞"，
    将它们视为同一个修复区域。
    
    Args:
        components: 连通分量列表
        tube_vertices: 管道顶点的坐标数组
        merge_distance_ratio: 合并距离阈值系数
    
    Returns:
        merged_components: 合并后的分量列表
    """
    if len(components) == 0:
        return components
    
    # 计算每个分量的中心点和尺寸
    n = len(components)
    centers = []
    sizes = []
    for comp in components:
        pts = tube_vertices[comp]
        center = pts.mean(axis=0)
        size = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        centers.append(center)
        sizes.append(size)
    centers = np.array(centers)
    sizes = np.array(sizes)
    
    print(f"  原始孔洞数量: {n}")
    
    # Union-Find 数据结构
    parent = list(range(n))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 构建合并图：距离过近则标记需合并
    should_merge_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            # 合并阈值：中心距离 < 两者尺寸的最大值 × 系数
            threshold = max(sizes[i], sizes[j]) * merge_distance_ratio
            if dist < threshold:
                union(i, j)
                should_merge_pairs.append((i, j, dist))
    
    if should_merge_pairs:
        print(f"  合并 {len(should_merge_pairs)} 对相邻孔洞")
        for i, j, d in should_merge_pairs[:5]:  # 只打印前5对
            print(f"    孔洞 {i} 和 {j} 距离 {d:.4f} 被合并")
        if len(should_merge_pairs) > 5:
            print(f"    ... 还有 {len(should_merge_pairs) - 5} 对")
    
    # 按合并后的根节点分组
    merged_groups = {}
    for i in range(n):
        root = find(i)
        if root not in merged_groups:
            merged_groups[root] = []
        merged_groups[root].append(i)
    
    # 合并每个组内的顶点索引
    result = []
    for group in merged_groups.values():
        merged_vertices = np.concatenate([components[i] for i in group])
        result.append(merged_vertices.astype(np.int32))
    
    print(f"  合并后孔洞数量: {len(result)}")
    
    return result


def analyze_mesh_texture_regions(mesh: o3d.geometry.TriangleMesh, sample_points=50000, curvature_percentile=80, cmap_name='turbo'):
    # 如果没有法向，估计法向
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # 点云采样 + 法向估计
    pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    # 曲率估计：用法向梯度近似
    normals = np.asarray(pcd.normals)
    diffs = np.linalg.norm(np.gradient(normals, axis=0), axis=1)
    curvature = diffs

    # 非线性增强
    curvature_enhanced = np.log1p(curvature)

    # 归一化 + colormap 映射
    norm = Normalize(vmin=curvature_enhanced.min(), vmax=curvature_enhanced.max())
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(norm(curvature_enhanced))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # # 可视化颜色图例
    # plt.figure(figsize=(8, 1.2))
    # plt.title("Curvature Colormap", fontsize=10)
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, orientation='horizontal', aspect=40, pad=0.2)
    # cbar.set_label('Enhanced Curvature Value', fontsize=9)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # 输出高曲率索引
    threshold = np.percentile(curvature, curvature_percentile)
    high_curv_idx = np.where(curvature >= threshold)[0]
    print(f"✅ 高曲率阈值（{curvature_percentile}%）：{threshold:.5f}")
    print(f"🎯 强纹理区域点数：{len(high_curv_idx)} / {sample_points}")

    return pcd, curvature_enhanced

def extract_holes_tube_mesh(ply_path, mesh_root, tube_radius=0.01):
    # 用 pyvista 加载网格，提取孔洞边界
    mesh_pv = pv.read(ply_path)
    meshfix = mf.MeshFix(mesh_pv)
    holes = meshfix.extract_holes()

    # 记录原始孔洞边界的顶点数和点数
    n_hole_pts = holes.n_points
    n_hole_cells = holes.n_cells
    print(f"  原始孔洞边界: {n_hole_pts} 个顶点, {n_hole_cells} 个单元")

    # 生成管状可视化版本
    tube = holes.tube(radius=tube_radius)
    tube.point_data["RGB"] = np.full((tube.n_points, 3), [255, 0, 0], dtype=np.uint8)
    
    # 保存为 ply
    tube.save(mesh_root+"hole_boundary_tube.ply")

    # 返回 open3d mesh（用于可视化）和原始孔洞 PolyData（包含 lines 信息用于检测）
    tube_o3d = o3d.io.read_triangle_mesh(mesh_root+"hole_boundary_tube.ply")
    return tube_o3d, holes

def merge_meshes(mesh_a: o3d.geometry.TriangleMesh, mesh_b: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # 合并两个TriangleMesh：顶点、法线、颜色、三角面
    va = np.asarray(mesh_a.vertices)
    vb = np.asarray(mesh_b.vertices)
    ta = np.asarray(mesh_a.triangles)
    tb = np.asarray(mesh_b.triangles)

    # 处理颜色
    has_color_a = mesh_a.has_vertex_colors()
    has_color_b = mesh_b.has_vertex_colors()
    ca = np.asarray(mesh_a.vertex_colors) if has_color_a else None
    cb = np.asarray(mesh_b.vertex_colors) if has_color_b else None

    if not has_color_a:
        ca = np.tile(np.array([[0.8, 0.8, 0.8]]), (va.shape[0], 1))
    if not has_color_b:
        cb = np.tile(np.array([[1.0, 0.0, 0.0]]), (vb.shape[0], 1))

    # 合并顶点与颜色
    v_all = np.vstack([va, vb]) if va.size and vb.size else (va if vb.size == 0 else vb)
    c_all = np.vstack([ca, cb]) if ca is not None and cb is not None else (ca if cb is None else cb)

    # 合并三角面（注意索引偏移）
    if ta.size and tb.size:
        tb_offset = tb + va.shape[0]
        t_all = np.vstack([ta, tb_offset])
    elif ta.size:
        t_all = ta
    elif tb.size:
        t_all = tb
    else:
        t_all = np.empty((0, 3), dtype=np.int32)

    merged = o3d.geometry.TriangleMesh()
    merged.vertices = o3d.utility.Vector3dVector(v_all)
    if c_all is not None:
        merged.vertex_colors = o3d.utility.Vector3dVector(c_all)
    if t_all.size:
        merged.triangles = o3d.utility.Vector3iVector(t_all)
    merged.compute_vertex_normals()
    return merged


def extract_mean_coords(mesh_root, timestamp):
    # 路径设置
    prefix = f'{int(timestamp):05d}'
    raw_mesh_path = mesh_root+f'{prefix}_mesh.ply'

    # 用 open3d 加载 mesh
    mesh_o3d = o3d.io.read_triangle_mesh(raw_mesh_path)

    # 分析曲率并获取彩色点云
    print("🔍 分析凹凸程度...")
    pcd, curvature_enhanced = analyze_mesh_texture_regions(mesh_o3d)

    # 提取孔洞边界（红色管道），同时获取 PyVista PolyData（用于正确检测孔洞）
    print("🔧 提取孔洞边界...")
    tube_o3d, tube_pv = extract_holes_tube_mesh(raw_mesh_path, mesh_root)
    tube_o3d.paint_uniform_color([0.5, 0.5, 0.5])  # 确保红色显示

    merged_mesh = merge_meshes(mesh_o3d, tube_o3d)

    # 保存合并网格
    o3d.io.write_triangle_mesh(mesh_root+"path_to_merged_model.ply", merged_mesh)
    print("✅ 合并后的网格已保存为 path_to_merged_model.ply")

    # 可视化：曲率颜色点云 + 红色孔洞标注
    print("👁️ 最终可视化...")
    # o3d.visualization.draw_geometries([pcd, tube_o3d], window_name="Curvature + Hole Visualization")

    picked_indices = []
    try:
        # =========================================================================
        # 孔洞检测与处理流程（基于 PyVista PolyData 的 lines 结构）
        # =========================================================================
        # 从 PolyData 中提取 lines 信息
        lines = tube_pv.lines
        if tube_pv.n_points == 0 or len(lines) == 0:
            raise RuntimeError("未检测到孔洞边界。")

        print(f"  孔洞 lines 数组长度: {len(lines)}, 顶点数: {tube_pv.n_points}")

        # 计算 mesh 的包围盒对角线（用于尺寸过滤）
        mesh_bbox_min = np.asarray(mesh_o3d.vertices).min(axis=0)
        mesh_bbox_max = np.asarray(mesh_o3d.vertices).max(axis=0)
        mesh_bbox_diagonal = np.linalg.norm(mesh_bbox_max - mesh_bbox_min)
        print(f"  Mesh 包围盒对角线: {mesh_bbox_diagonal:.4f}")

        # =========================================================================
        # 步骤1: 从 PolyData lines 中解析每个独立的闭合环（一个孔洞 = 一个环）
        # PolyData lines 格式: [n0, v0, v1, ..., n1, v0, v1, ...]
        # 其中每个 line 由 n_i+1 个整数描述：n_i 是顶点数，后跟 n_i 个顶点索引
        # =========================================================================
        # 构建顶点邻接表（基于 lines 而非 triangles）
        n_points = tube_pv.n_points
        adjacency = [[] for _ in range(n_points)]
        idx = 0
        while idx < len(lines):
            n_verts = lines[idx]
            idx += 1
            for j in range(n_verts):
                v = int(lines[idx + j])
                if j > 0:
                    prev = int(lines[idx + j - 1])
                    adjacency[prev].append(v)
                    adjacency[v].append(prev)
                if j == n_verts - 1 and n_verts > 2:
                    # 闭合环：首尾相连
                    first = int(lines[idx])
                    adjacency[v].append(first)
                    adjacency[first].append(v)
            idx += n_verts

        # BFS 找连通分量（每个分量 = 一个闭合管状环 = 一个孔洞）
        visited = np.zeros(n_points, dtype=bool)
        components = []
        for v in range(n_points):
            if visited[v]:
                continue
            queue = [v]
            visited[v] = True
            comp = []
            while queue:
                cur = queue.pop()
                comp.append(cur)
                for nb in adjacency[cur]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if comp:
                components.append(np.array(comp, dtype=int))

        print(f"🔍 检测到 {len(components)} 个初始连通分量（孔洞环）")
        for i, comp in enumerate(components):
            print(f"   环 {i}: 顶点数={len(comp)}")

        # 获取管道的顶点坐标（来自 PyVista PolyData，与 line 索引对应）
        tube_vertices = tube_pv.points

        # =========================================================================
        # 步骤2: 过滤掉过长的边界
        # =========================================================================
        print("🧹 过滤过长的边界...")
        filtered_components = filter_boundary_holes(components, tube_vertices, mesh_bbox_diagonal)

        if len(filtered_components) == 0:
            raise RuntimeError("过滤后没有剩余孔洞。")

        # =========================================================================
        # 步骤3: 找到最大的孔洞（不合并）
        # =========================================================================
        def bbox_diag(points: np.ndarray) -> float:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            return float(np.linalg.norm(maxs - mins))

        hole_sizes = []
        for comp in filtered_components:
            pts = tube_vertices[comp]
            hole_sizes.append(bbox_diag(pts))

        largest_hole_idx = int(np.argmax(hole_sizes))
        largest_hole_size = hole_sizes[largest_hole_idx]
        largest_hole_comp = filtered_components[largest_hole_idx]
        largest_hole_points = tube_vertices[largest_hole_comp]

        print(f"🎯 最大孔洞: 索引={largest_hole_idx}, 尺寸={largest_hole_size:.4f}, 顶点数={len(largest_hole_comp)}")

        # =========================================================================
        # 步骤4: 在最大孔洞边界上均匀选择4个点
        # =========================================================================
        hole_center = largest_hole_points.mean(axis=0)

        # 估计孔洞的局部切平面
        centered = largest_hole_points - hole_center
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        normal_vec = evecs[:, 0]
        tangent1 = evecs[:, 2]
        tangent2 = np.cross(normal_vec, tangent1)

        # 归一化
        normal_vec = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)
        tangent1 = tangent1 / (np.linalg.norm(tangent1) + 1e-12)
        tangent2 = tangent2 / (np.linalg.norm(tangent2) + 1e-12)

        # 投影到切平面
        proj_x = centered @ tangent1
        proj_y = centered @ tangent2
        angles = np.arctan2(proj_y, proj_x)
        radii = np.sqrt(proj_x ** 2 + proj_y ** 2)

        # 设定四个扇区中心角：0, 90, 180, -90 度
        sector_centers = np.array([0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi])
        sector_width = 0.5 * np.pi  # 每个扇区宽 90°

        chosen = []
        for theta in sector_centers:
            ang_diff = np.abs(np.arctan2(np.sin(angles - theta), np.cos(angles - theta)))
            in_sector = ang_diff <= (sector_width * 0.5)
            cand_ids = np.where(in_sector)[0]
            if cand_ids.size == 0:
                # 如果扇区为空，选择角度最近的点
                cand_ids = np.array([int(np.argmin(ang_diff))])
            else:
                # 选择距离分布最平均的点（距离中心适中，避免最内或最外）
                radii_cand = radii[cand_ids]
                # 用距离中心百分比作为选择标准
                radii_norm = radii_cand / (radii_cand.max() + 1e-12)
                # 选择距离适中的点（0.3-0.7范围），如果没有则选最远的
                mid_mask = (radii_norm >= 0.3) & (radii_norm <= 0.7)
                if mid_mask.any():
                    mid_ids = cand_ids[mid_mask]
                    cand_ids = np.array([mid_ids[int(len(mid_ids) // 2)]])
                else:
                    cand_ids = np.array([cand_ids[int(np.argmax(radii_cand))]])

            chosen.append(int(largest_hole_comp[cand_ids[0]]))

        picked_indices = list(dict.fromkeys(chosen))[:4]

        # 如果不够4个，补充
        if len(picked_indices) < 4:
            remaining = [int(i) for i in largest_hole_comp if int(i) not in picked_indices]
            for rid in remaining:
                if len(picked_indices) >= 4:
                    break
                picked_indices.append(rid)

        print(f"✅ 最终选择的4个点索引: {picked_indices}")
        print(f"   对应坐标:")
        for idx in picked_indices:
            print(f"     索引 {idx}: {tube_vertices[idx]}")

        # 计算偏移量：在合并后的 mesh 中，tube 顶点的索引需要加上原始 mesh 的顶点数
        offset = np.asarray(mesh_o3d.vertices).shape[0]
        picked_indices_with_offset = [int(idx + offset) for idx in picked_indices]
        print(f"   合并mesh中的索引（偏移={offset}）: {picked_indices_with_offset}")

        # 保存原始索引和坐标用于可视化
        original_indices = picked_indices
        original_coords = tube_vertices[picked_indices].copy()

        # =========================================================================
        # 可视化：叠加4个小球标记
        # =========================================================================
        if len(original_indices) == 4:
            try:
                marker_colors = [
                    [1.0, 0.0, 0.0],  # 红
                    [0.0, 1.0, 0.0],  # 绿
                    [0.0, 0.0, 1.0],  # 蓝
                    [1.0, 1.0, 0.0],  # 黄
                ]
                marker_radius = max(0.003, min(0.02, 0.02))
                markers = []
                # 使用原始坐标进行可视化，避免索引问题
                for i, pt in enumerate(original_coords):
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
                    sphere.compute_vertex_normals()
                    sphere.paint_uniform_color(marker_colors[i % len(marker_colors)])
                    sphere.translate(pt)
                    markers.append(sphere)
                print(f"👁️ 展示4个选中点标记（半径={marker_radius:.4f}）...")
                # o3d.visualization.draw_geometries([pcd, tube_o3d] + markers, window_name="Curvature + Hole Visualization + Picks")
            except Exception as vis_e:
                print(f"标记点可视化失败：{vis_e}")

        # 返回带 offset 的索引（用于合并 mesh 中的顶点索引）
        picked_indices = picked_indices_with_offset
    except Exception as e:
        print(f"自动选择4个点失败：{e}")
        picked_indices = []

    return picked_indices


# def extract_holes(mesh_root, timestamp):
#     # 读取PLY文件中的mesh
#     prefix = f'{int(timestamp):05d}'
#     mesh_file = mesh_root+f'{prefix}_mesh.ply'
#     mesh = pv.read(mesh_file)

#     # 使用pymeshfix进行孔洞提取
#     meshfix = mf.MeshFix(mesh)
#     holes = meshfix.extract_holes()
#     # 可视化原始mesh和孔洞
#     p = pv.Plotter()
#     p.add_mesh(mesh, color=True)
#     p.add_mesh(holes, color="r", line_width=8)
#     p.enable_eye_dome_lighting()
#     # p.show()
#     # 将孔洞边界线转换为红色管状结构
#     tube_radius = 0.01  # 管状结构的半径，根据实际情况调整
#     tube = holes.tube(radius=tube_radius)
#     tube.point_data["RGB"] = np.full((tube.n_points, 3), [255, 0, 0], dtype=np.uint8)  # 设置为红色
#     # 保存为PLY文件
#     tube.save(mesh_root+"extract_holes.ply")

#     # 加载模型
#     mesh1 = trimesh.load(mesh_file)
#     mesh2 = trimesh.load(mesh_root+'extract_holes.ply')
#     # 合并模型
#     merged_mesh = trimesh.util.concatenate([mesh1, mesh2])
#     merged_mesh.export(mesh_root+"path_to_merged_model.ply")


# class PointPickingVisualizer:
#     def __init__(self, point_cloud):
#         self.point_cloud = point_cloud
#         self.picked_indices = []  # 存储选中的点索引

#     def run(self):
#         picked_indices = self.pick_points()
#         assert len(picked_indices) >= 4 or len(picked_indices) == 0
#         if picked_indices:
#             for idx in picked_indices:
#                 self.picked_indices.append(idx)
#                 point = np.asarray(self.point_cloud.points)[idx]
#             if len(self.picked_indices) >= 4:
#                 print("4 points selected. Exiting...")
#         return self.picked_indices
        

#     def pick_points(self):
#         print("Press at least select 4 points by [shift+left]. Cacel by [shift+right]. Close by [Q]")
#         # vis = o3d.visualization.VisualizerWithKeyCallback()
#         vis = o3d.visualization.VisualizerWithEditing()
#         vis.create_window()
#         vis.add_geometry(self.point_cloud)
#         # vis.register_key_callback(ord("S"), self.pick_points_callback)  # S键触发点选逻辑
#         vis.run()
#         vis.destroy_window()

#         return vis.get_picked_points()

    # def run(self):
    #     picked_indices = self.pick_points()
    #     assert len(picked_indices) >= 4 or len(picked_indices) == 0
    #     if picked_indices:
    #         for idx in picked_indices:
    #             self.picked_indices.append(idx)
    #             point = np.asarray(self.point_cloud.points)[idx]
    #             print(f"Picked point index: {idx}, coordinates: {point}")
    #         if len(self.picked_indices) >= 4:
    #             print("4 points selected. Exiting...")
    #     return self.picked_indices

    # def pick_points(self):
    #     print("Please select at least 4 points using [shift + left click], then press [Q] to finish.")
    #     print("point_cloud:", self.point_cloud)
    #     picked_indices = o3d.visualization.draw_geometries_with_editing([self.point_cloud])
    #     print("picked_indices done!")
    #     return picked_indices


def culculate_extrinsic(mesh_root, timestamp, diffusion_idx):
    # # 1. 加载 .ply 文件
    # extract_holes(mesh_root, timestamp)
    picked_indices = extract_mean_coords(mesh_root, timestamp)
    # 读取合并网格的采样点云便于索引
    point_cloud = o3d.io.read_point_cloud(mesh_root+"path_to_merged_model.ply")

    # # 2. 初始化点选工具
    # picker = PointPickingVisualizer(point_cloud)
    # # 3. 运行交互工具并获取选中点
    # picked_indices = picker.run()
    # # 4. 输出选中的点坐标
    if picked_indices:
        points = np.asarray(point_cloud.points)
        selected_coords = points[picked_indices]
        print(f"Selected point indices: {picked_indices}")
        print(f"Selected point coordinates:\n{selected_coords}")
        mean_coords = np.mean(selected_coords, axis=0)
        print(f"mean_coords:\n{mean_coords}")
    else:
        print("No points selected.")
        return None
    
    # 加载合并后的Mesh
    mesh = o3d.io.read_triangle_mesh(mesh_root+"path_to_merged_model.ply")
    # 计算顶点法线
    mesh.compute_vertex_normals()
    # 获取顶点法线
    vertex_normals = np.asarray(mesh.vertex_normals)
    # 指定需要输出的顶点索引（例如：顶点索引 0, 2, 5）
    indices_to_output = picked_indices
    # 输出指定顶点的法线
    for idx in indices_to_output:
        print(f"Vertex {idx} normal: {vertex_normals[idx]}")
    # 计算这些指定法线的均值
    selected_normals = vertex_normals[indices_to_output]
    # 求均值
    mean_normal = np.mean(selected_normals, axis=0)
    # 输出法线均值
    print(f"Mean normal of selected vertices: {mean_normal}")

    # 计算前进后的点的坐标
    mean_coords = np.array(mean_coords)
    mean_normal = np.array(mean_normal)
    distance = 0.3  # 前进的距离（1）
    new_point = mean_coords + mean_normal * distance
    # 输出结果
    print("前进后的点坐标:", new_point)

    # 已知的平移向量（相机位置）
    t = np.array(new_point)
    # 已知的法线向量（相机平面的法线）
    n = np.array(mean_normal)*(-1)
    # 归一化法线向量
    n_normalized = n / np.linalg.norm(n)
    # 假设世界坐标系中的某一轴是Y轴，我们可以用它来计算x轴和y轴
    # 假设在这个例子中，世界坐标系的y轴是[0, 1, 0]
    y_world = np.array([0, 1, 0])
    # 计算x轴方向（相机坐标系中的x轴），使用叉积
    x_camera = np.cross(y_world, n_normalized)
    # 归一化x轴
    x_camera = x_camera / np.linalg.norm(x_camera)
    # 计算y轴方向（相机坐标系中的y轴）
    y_camera = np.cross(n_normalized, x_camera)
    # 构造旋转矩阵
    R = np.column_stack((x_camera, y_camera, n_normalized))
    # 构建外参矩阵 [R | t]
    extrinsic_matrix = np.column_stack((R, t))
    # 输出外参矩阵
    print("Extrinsic Matrix:")
    print(extrinsic_matrix)

    # 定义外参矩阵
    extrinsic_matrix = np.array(extrinsic_matrix)
    # 保存为 .npy 文件
    pose_file = f"{diffusion_idx}.npy"  # 使用索引作为文件名
    pose_path = os.path.join("diffusion_views", pose_file)
    np.save(pose_path, extrinsic_matrix)
    print("外参矩阵已保存为", pose_path)

    return extrinsic_matrix


def run_diffusion(diffusion_idx, diffusion_num):
    print("run diffusion")
    diffusion_num1 = diffusion_num
    files = os.listdir("diffusion_views")
    # 遍历文件并复制、重命名
    for file in files:
        # 获取文件名和扩展名
        filename, extension = os.path.splitext(file)
        # 生成新的文件名（三位数字格式）
        new_filename = f"{int(filename):03d}{extension}"
        
        # 构造源文件路径和目标文件路径
        source_path = os.path.join("diffusion_views", file)
        target_path = os.path.join("EscherNet/demo/GSO30/endo/render_mvs_25/model", new_filename)
        # 复制并重命名文件
        shutil.copy(source_path, target_path)
        
        if filename in diffusion_idx:
            if not os.path.exists("diffusion_views_fix"):
                os.makedirs("diffusion_views_fix")
            new_filename = f"{diffusion_num1}{extension}"
            target_path_fix = os.path.join("diffusion_views_fix", new_filename)
            shutil.copy(source_path, target_path_fix)
            diffusion_num1 += 1

    # 获取当前conda环境的路径
    conda_path = subprocess.check_output(["which", "conda"]).decode().strip()

    # 构造激活env2环境并运行命令的字符串
    # 假设要在env2环境中运行一个Python脚本script_in_env2.py
    command = f"{conda_path} run -n eschernet bash eval_eschernet.sh"

    # 指定新的工作目录
    new_working_directory = "EscherNet/"

    # 使用subprocess.run创建子进程，并指定工作目录
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=new_working_directory)
    if result.returncode == 0:
        print("命令成功执行！")
        print("输出内容：")
        print(result.stdout)
    else:
        print("命令执行失败！")
        print("错误信息：")
        print(result.stderr)
    # 可以获取子进程的输出等信息
    print(result.stdout)

    files = os.listdir("EscherNet/logs_6DoF/GSO25/N5M25/endo")
    # 遍历文件并复制、重命名
    for file in files:
        # 构造源文件路径和目标文件路径
        source_path = os.path.join("EscherNet/logs_6DoF/GSO25/N5M25/endo", file)
        target_path = os.path.join("diffusion_views", file)
        # 复制并重命名文件
        shutil.copy(source_path, target_path)
        
        filename, extension = os.path.splitext(file)
        if filename in diffusion_idx:
            new_filename = f"{diffusion_num}{extension}"
            target_path_fix = os.path.join("diffusion_views_fix", new_filename)
            shutil.copy(source_path, target_path_fix)
            diffusion_num += 1
            
    return diffusion_num
