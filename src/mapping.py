import os
import cv2
import numpy as np
import random
import torch
torch.autograd.set_detect_anomaly(True)
from colorama import Fore, Style
from torch.autograd import Variable
from .Logger import TextLogger
from .nerf_func import random_select, build_rays, Rt_to_quaternion, quaternion_to_Rt
import matplotlib.pyplot as plt
from time import gmtime, strftime, time, sleep
from .diffusion_views import culculate_extrinsic, run_diffusion
from PIL import Image
import shutil

class Mapper(object):
    """
    Mpper thread.
    """

    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args
        self.verbose = slam.verbose

        self.bound = slam.bound
        self.video = slam.video
        self.mapping_net = slam.mapping_net
        self.ConsistNet = slam.ConsistNet
        self.renderer = slam.renderer
        self.reload_map = slam.reload_map
        self.diffusion_num = 0

        self.output = slam.output

        self.device = cfg['mapping']['device']
        self.num_joint_iters = cfg['mapping']['iters']
        self.decay = float(cfg['mapping']['decay'])
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.w_sdf_loss = cfg['mapping']['w_sdf_loss']
        self.w_eikonal_loss = cfg['mapping']['w_eikonal_loss']
        self.uncertainty_based = cfg['mapping']['uncertainty_weight_loss']

        self.BA = cfg['mapping']['BA']  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.local_step = 0
        self.global_step = 0
        self.last_visit = 0
        self.init = True

        os.makedirs(f'{self.output}/logs/mapping/', exist_ok=True)
        self.txt = TextLogger(f'{self.output}/logs/mapping/log.txt')

        ignore_keys = ()
        net_param = self.mapping_net.get_training_parameters(ignore_keys=ignore_keys)
        grid_param = self.mapping_net.get_volume_parameters()
        self.train_params = list(net_param) + list(grid_param)
        self.optimizer = torch.optim.AdamW([
            {'params': net_param, 'lr': cfg['mapping']['net_lr']},
            {'params': grid_param, 'lr': cfg['mapping']['grid_lr']},
        ], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        
        # self.ucnerf_initialized = False

    
    def get_diffusion_items(self, folder_path, image_size, device, diffusion_idx):
        """
        从文件夹中生成所需的字典，所有矩阵转换为PyTorch张量
        :param folder_path: 包含图像和npy文件的文件夹路径
        :param image_size: 图像resize后的大小 (h, w)
        :param device: 设备（'cpu' 或 'cuda'）
        :return: 字典，key是索引，value是包含image, depth, c2w, gt_c2w, mask的字典
        """
        result_dict = {}
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, image_file in enumerate(image_files):
            # 获取文件名（不带扩展名）
            base_name = os.path.splitext(image_file)[0]
            if base_name in diffusion_idx:
                # 读取图像并resize
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (image_size[1], image_size[0]))  # (w, h)
                image = torch.from_numpy(image).float().to(device)  # 转换为torch张量，形状为[h, w, 3]
                image = image / 255.0
                
                # 读取对应的npy文件并转换为4x4矩阵
                npy_path = os.path.join(folder_path, f"{base_name}.npy")
                if not os.path.exists(npy_path):
                    raise FileNotFoundError(f"No npy file found for {base_name}")
                c2w = np.load(npy_path)
                c2w = np.vstack([c2w, [0, 0, 0, 1]])  # 添加最后一行
                c2w = torch.from_numpy(c2w).float().to(device)  # 转换为torch张量，形状为[4, 4]
                
                # 创建随机深度图
                depth = torch.ones(image_size[0], image_size[1], device=device, dtype=torch.float)*0.3 # 随机深度图，形状为[h, w]
                
                # 创建全1的mask
                mask = torch.ones(image_size[0], image_size[1], device=device, dtype=torch.float)  # 全1掩码，形状为[h, w]
                
                # 创建字典
                result_dict[idx] = image, depth, c2w, c2w.clone(), mask
        
        return result_dict
    
    def save_reference_items(self, folder_path, image_size, image, c2w, idx):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # 从torch张量恢复图像
        image = image.cpu().numpy()  # 转换为numpy数组
        image = (image * 255).astype(np.uint8)  # 恢复到[0, 255]范围
        image_pil = Image.fromarray(image)

        # 转换为RGBA格式（添加透明通道）
        image_rgba = image_pil.convert('RGBA')
        # 如果需要调整图像大小，可以使用Pillow的resize方法
        image_rgba = image_rgba.resize((image_size[1], image_size[0]), Image.ANTIALIAS)

        # 保存图像文件
        image_file = f"{idx}.png"  # 使用索引作为文件名
        image_path = os.path.join(folder_path, image_file)
        image_rgba.save(image_path)

        # 从torch张量恢复c2w矩阵
        c2w = c2w.cpu().numpy()  # 转换为numpy数组
        c2w = c2w[:3, :]  # 去掉最后一行，恢复为3x4矩阵

        # 保存npy文件
        npy_file = f"{idx}.npy"  # 使用索引作为文件名
        npy_path = os.path.join(folder_path, npy_file)
        np.save(npy_path, c2w)


    def optimize_map(self,
                     rays_o,
                     rays_d,
                     rays_color,
                     rays_depth,
                     optimizer,
                     num_joint_iters,
                     volume_feature=None,
                     uncertainty_map=None,
                     mvs_depth=None,
                     outputs=None,
                     pose_ref=None,
                     imgs_input=None,
                     near_far=None,
                     near_fars=None):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enables).
        Args:
            num_joint_iters:            (int), number of mapping iterations.
            lr_factor:                  (float), current_lr * lr_factor
            writer:                     Tensorboard SummaryWriter

        Returns:
            cur_c2w/None:               (Tensor), the updated cur_c2w, return None if no BA

        """
        device = self.device

        for joint_iter in range(1, num_joint_iters+1):
            self.local_step += 1
            self.global_step += 1

            optimizer.zero_grad()

            render_params = {
                'global_step': self.global_step,
            }
            uncertainty_params = {
                'volume_feature': volume_feature,
                'uncertainty_map': uncertainty_map,
                'mvs_depth': mvs_depth,
                'outputs': outputs,
                'pose_ref': pose_ref,
                'imgs_input': imgs_input,
                'near_fars': near_fars
            }

            with torch.enable_grad():
                ret = self.renderer.render_batch_ray(rays_o=rays_o, rays_d=rays_d, net=self.mapping_net.to(device),
                                                     render_params=render_params, device=device, gt_depth=rays_depth,
                                                     uncertainty_params=uncertainty_params, near_far=near_far)

            rays_depth = rays_depth.reshape(-1, 1)  # [n_rays, 1]
            valid_mask = (rays_depth > 0).reshape(-1)  # [n_rays, ]
            mvs_depth = mvs_depth.reshape(-1, 1)

            rays_depth = rays_depth[valid_mask]
            mvs_depth = mvs_depth[valid_mask]
            rays_color = rays_color[valid_mask]
            est_color = ret['color'][valid_mask]  # [n_rays, 3]
            est_depth = ret['depth'][valid_mask]  # [n_rays, 1]
            sdf = ret['sdf'][valid_mask]  # [n_rays, n_samples]
            z_vals = ret['z_vals'][valid_mask]  # [n_rays, n_samples]
            depth_variance = ret['depth_variance'][valid_mask] # [n_rays, 1]
            gradient_error = ret['gradient_error']  # [1, ]
            uncertainty_weight = 1.0 / torch.sqrt(depth_variance.detach() + 1e-10) # [n_rays, 1]
            if not self.uncertainty_based:
                uncertainty_weight = torch.ones_like(uncertainty_weight)

            assert rays_depth.shape == est_depth.shape, f'{rays_depth.shape}, {est_depth.shape}!'
            assert rays_depth.shape == mvs_depth.shape, f'{rays_depth.shape}, {mvs_depth.shape}!'

            total_loss = 0.0

            # -- color loss --
            color_loss = torch.abs(est_color - rays_color).mean()
            total_loss = total_loss + color_loss * self.w_color_loss

            # -- depth loss --
            mae = torch.abs(est_depth - rays_depth) + torch.abs(mvs_depth - rays_depth)
            depth_loss = (mae * uncertainty_weight).mean()
            total_loss = total_loss + depth_loss * 1.0

            # -- sdf loss --
            sdf_loss, sparse_loss = 0.0, 0.0
            if self.w_sdf_loss > 0:
                sdf_loss, sparse_loss = self.mapping_net.compute_sdf_error(sdf=sdf, z_vals=z_vals, gt_depth=rays_depth)
                total_loss = total_loss + (sdf_loss + sparse_loss) * self.w_sdf_loss

            # # -- eikonal loss --
            # if self.w_eikonal_loss > 0:
            #     eikonal_loss = gradient_error.mean()
            #     total_loss = total_loss + self.w_eikonal_loss * eikonal_loss

            total_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.train_params, max_norm=35.0)
            optimizer.step()
            optimizer.zero_grad()

            if (self.local_step % self.num_joint_iters == 0) and self.verbose:
                msg = ''
                list_lr = []
                for g in optimizer.param_groups:
                    list_lr.append(round(g['lr'], 6))
                msg += 'Lr : {}'.format(list_lr)
                msg += f' | Loss of total: {total_loss.detach():.4f}, depth: {depth_loss:.4f}, ' \
                       f'color: {color_loss:.4f}, ' \
                       f'sdf: {sdf_loss:.4f}, n_rays: {rays_o.shape}!'
                self.txt.info(msg)

    def unpreprocess(self, data, shape=(1, 1, 3, 1, 1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224,
                             -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224,
                            1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std

    # def ensure_ucnerf(self):
    #     if not self.ucnerf_initialized:
    #         self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_ucnerf(
    #             self.cfg, dir_embedder=True, pts_embedder=True)
    #         self.Consist_Learner = self.render_kwargs_train['network_mvs']
    #         filter_keys(self.render_kwargs_train)
    #         self.render_kwargs_train.pop('network_mvs')
    #         self.render_kwargs_train['NDC_local'] = False
    #         self.ucnerf_initialized = True
    
    def get_ucnerf_rays(self, frame_indices, unvisit_frame, H, W, fx, fy, cx, cy):
        from .utils.utils import convert_frames_to_ucnerf, build_rays_ucnerf, process_gt_depth_for_ucnerf
        print("frame_indices",frame_indices)
        gt_color, gt_depth, c2w, gt_c2w, mask = unvisit_frame[frame_indices[0]]
        # convert input for UC-NeuS
        imgs, imgs_input, affine_mat, affine_mat_inv, near_far, pose_ref = convert_frames_to_ucnerf(  
            unvisit_frame, frame_indices, fx, fy, cx, cy, H, W
        )
        volume_feature, uncertainty_map, mvs_depth, outputs = self.ConsistNet(
            imgs_input,
            affine_mat,
            affine_mat_inv,
            near_far,
            pad=0)
        imgs = self.unpreprocess(imgs)
        # print("imgs shape:",imgs.shape)
        # print("mvs_depth shape:",mvs_depth.shape)
        
        # print(f"Uncertainty map shape: {uncertainty_map.shape}")
        uncertainty_map = self.mapping_net.forward_uncertainty(uncertainty_map.reshape(1, -1, 1)).reshape(H, W)
        # plt.imshow(uncertainty_map.cpu().detach().numpy(), cmap='viridis')  
        # plt.colorbar()  
        # plt.title('Uncertainty Map')  
        # plt.savefig(f'{self.output}/logs/mapping/uncertainty_map.png')
        # plt.close()

        # print(f"gt_depth: {gt_depth.shape}",gt_depth.min(),gt_depth.max())
        sparse_depths, coords, sparse_depths_ms, rays_depth = process_gt_depth_for_ucnerf(gt_depth, img_wh=(W, H))
        # print(f"sparse_depths: {sparse_depths.shape}",f"coords: {coords.shape}")
        rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_o, rays_depth, ndc_parameters, pixel_coordinates, near_fars = \
            build_rays_ucnerf(patch_num=50, patch_size=6, imgs=imgs, mvs_confidence=uncertainty_map, sparse_depths=sparse_depths, coords=coords, pose_ref=pose_ref, w2cs=pose_ref['w2cs'], c2ws=pose_ref['c2ws'], intrinsics=pose_ref['intrinsics'],\
                N_rays=2000, N_samples=72, pad=0, with_depth=True, outputs=outputs)
        
        unvisit_rays_o = rays_o
        unvisit_rays_d = rays_dir
        
        # print(f"pixel_coordinates shape: {pixel_coordinates.shape}")
        unvisit_gt_depth = sparse_depths[pixel_coordinates[0, :], pixel_coordinates[1, :]]
        unvisit_gt_color = target_s
        
        # 对mvs_depth进行类似处理
        mvs_depth_sparse = mvs_depth[0, pixel_coordinates[0, :], pixel_coordinates[1, :]]  # 去掉batch维度并索引
        # print(f"unvisit_rays_o shape: {unvisit_rays_o.shape}")
        # print(f"unvisit_rays_d shape: {unvisit_rays_d.shape}")
        # print(f"unvisit_gt_depth shape: {unvisit_gt_depth.shape}")
        # print(f"unvisit_gt_color shape: {unvisit_gt_color.shape}")
        return unvisit_rays_o, unvisit_rays_d, unvisit_gt_depth, unvisit_gt_color, volume_feature, uncertainty_map, mvs_depth_sparse, outputs, imgs, pose_ref, near_far, near_fars

        

    def __call__(self, the_end=False, iter=-1):
        # self.ensure_ucnerf()
        cur_idx = int(self.video.filtered_id.item())  # valid idx [0, 1, ..., cur_idx-1]
        if cur_idx > 1:
            # cur_idx = min(cur_idx, self.last_visit+per_keyframe)
            timestamp = self.video.timestamp[cur_idx-1]
            # print("timestamp", timestamp)
            num_joint_iters = self.num_joint_iters
            if the_end:
                num_joint_iters = num_joint_iters * 10
            device = self.device
            self.local_step = 0

            unvisit_list = list(range(self.last_visit, cur_idx))
            visit_list = [cur_idx-1, cur_idx-2]
            if self.last_visit > 0:
                priority = self.video.update_priority[:self.last_visit].detach()
                _, indices = torch.sort(priority, dim=0, descending=True)
                indices = list(indices.cpu().numpy())
                visit_list += indices[:10]
                visit_list += random_select(self.last_visit, self.mapping_window_size-12)

            visit_frame = {}
            visit_ba_list = []
            enable_ba = ((self.BA) and (self.last_visit >= 10))
            for frame_id, frame in enumerate(visit_list):
                frame_items = self.video.get_mapping_item(frame, device, decay=self.decay)
                visit_frame[frame] = frame_items
                if enable_ba:
                    _, _, c2w_mat, _, _ = frame_items
                    quadt = Rt_to_quaternion(c2w_mat, Tquad=False)
                    quadt = Variable(quadt.to(self.device), requires_grad=True)
                    visit_ba_list.append(quadt)


            unvisit_frame = {}
            for frame in unvisit_list:
                unvisit_frame[frame] = self.video.get_mapping_item(frame, device, decay=self.decay)

            H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

            optimizer = self.optimizer
            if enable_ba and len(optimizer.param_groups) > 2:  # Attention the number 2 set here
                del optimizer.param_groups[-1]
            if enable_ba and len(visit_ba_list) > 0:
                optimizer.add_param_group({'params': visit_ba_list, 'lr': self.BA_cam_lr})

            bd = self.video.get_bound()
            with self.video.mapping.get_lock():
                self.mapping_net.update_bound(bd)

            prefix = f"Bound: ["
            bd = self.mapping_net.realtime_bound.tolist()
            prefix += f'[{bd[0][0]:.1f}, {bd[0][1]:.1f}], '
            prefix += f'[{bd[1][0]:.1f}, {bd[1][1]:.1f}], '
            prefix += f'[{bd[2][0]:.1f}, {bd[2][1]:.1f}]]; '
            print(Fore.MAGENTA)
            if self.verbose:
                self.txt.info(prefix + 'Mapping Frame {}, unvisit {}, has visited {}'.format(timestamp.item(), unvisit_list, visit_list))
            else:
                if len(unvisit_list) > 2:
                    self.txt.info(prefix + 'Mapping Frame {}, unvisit kf are: {}!'.format(timestamp.item(), unvisit_list))
            print(Style.RESET_ALL)


            # unvisit
            unvisit_factor = num_joint_iters * 10 if self.init else num_joint_iters
            if len(unvisit_list) > 3: # 2
                self.last_visit = cur_idx
                for _ in range(unvisit_factor):
                    unvisit_rays_d = []
                    unvisit_rays_o = []
                    unvisit_gt_depth = []
                    unvisit_gt_color = []

                    # sub_unvisit_list = []
                    # if len(unvisit_list) < self.mapping_window_size:
                    #     sub_unvisit_list = unvisit_list
                    # else:
                    sub_unvisit_list = list(np.random.choice(unvisit_list, self.mapping_window_size))
                    n_rays_unvisit = self.mapping_pixels // len(sub_unvisit_list)

                    frame_indices = []
                    gt_depth = None
                    for frame_idx, frame in enumerate(sub_unvisit_list):
                        gt_color, gt_depth, c2w, gt_c2w, mask = unvisit_frame[frame]
                        if gt_depth.min() > 10.0:
                            # print("gt_depth",gt_depth.min())
                            continue
                        else:
                            # print("gt_depth",gt_depth.min())
                            other_indices = [f for i, f in enumerate(sub_unvisit_list) if i != frame_idx]
                            num_to_sample = min(3, len(other_indices))
                            sampled = random.sample(other_indices, num_to_sample)
                            frame_indices = [frame] + sampled 
                            unvisit_rays_o, unvisit_rays_d, unvisit_gt_depth, unvisit_gt_color, volume_feature, uncertainty_map, mvs_depth, outputs, imgs, pose_ref, near_far, near_fars = \
                                self.get_ucnerf_rays(frame_indices, unvisit_frame, H, W, fx, fy, cx, cy)
                            frame_indices = []

                            if len(unvisit_rays_o) < 100:
                                continue

                            self.optimize_map(
                                rays_o=unvisit_rays_o,
                                rays_d=unvisit_rays_d,
                                rays_color=unvisit_gt_color,
                                rays_depth=unvisit_gt_depth,
                                volume_feature=volume_feature,
                                uncertainty_map=uncertainty_map,
                                mvs_depth=mvs_depth,
                                outputs=outputs,
                                imgs_input=imgs[:, 1:],
                                pose_ref=pose_ref,
                                near_far=near_far,
                                near_fars=near_fars,
                                optimizer=optimizer,
                                num_joint_iters=1,
                            )

            torch.cuda.empty_cache()

            # visit
            for _ in range(num_joint_iters):
                if len(visit_list) < 4: # 1
                    continue
                self.diffusion_views = False
                n_rays = self.mapping_pixels // len(visit_list)
                visit_rays_d_list = []
                visit_rays_o_list = []
                visit_gt_depth_list = []
                visit_gt_color_list = []
                for frame_id, frame in enumerate(visit_list):
                    gt_color, gt_depth, c2w, gt_c2w, mask = visit_frame[frame]
                    if enable_ba:
                        quadt = visit_ba_list[frame_id]
                        c2w = quaternion_to_Rt(quadt)
                    if self.reload_map > 1000 and the_end:
                        folder_path = "diffusion_views"
                        self.save_reference_items(folder_path=folder_path, image_size=(H, W), image=gt_color, c2w=c2w, idx=frame_id)

                sub_visit_list = []
                if len(visit_list) < 13:
                    sub_visit_list = visit_list
                else:
                    sub_visit_list = list(np.random.choice(visit_list, 13))
                frame_indices = []
                gt_depth = None
                for frame_idx, frame in enumerate(sub_visit_list):
                    gt_color, gt_depth, c2w, gt_c2w, mask = visit_frame[frame]
                    if gt_depth.min() > 10.0:
                        # print("gt_depth",gt_depth.min())
                        continue
                    else:
                        # print("gt_depth",gt_depth.min())
                        other_indices = [f for i, f in enumerate(sub_visit_list) if i != frame_idx]
                        num_to_sample = min(3, len(other_indices))
                        sampled = random.sample(other_indices, num_to_sample)
                        frame_indices = [frame] + sampled 
                        visit_rays_o, visit_rays_d, visit_gt_depth, visit_gt_color, volume_feature, uncertainty_map, mvs_depth, outputs, imgs, pose_ref, near_far, near_fars = \
                            self.get_ucnerf_rays(frame_indices, visit_frame, H, W, fx, fy, cx, cy)
                        frame_indices = []

                        if len(visit_rays_o) < 100:
                            continue

                        if self.diffusion_views:
                            num_joint_iters_ = 1
                        else:
                            num_joint_iters_ = 1
                        self.optimize_map(
                            rays_o=visit_rays_o,
                            rays_d=visit_rays_d,
                            rays_color=visit_gt_color,
                            rays_depth=visit_gt_depth,
                            volume_feature=volume_feature,
                            uncertainty_map=uncertainty_map,
                            mvs_depth=mvs_depth,
                            outputs=outputs,
                            imgs_input=imgs[:, 1:],
                            pose_ref=pose_ref,
                            near_far=near_far,
                            near_fars=near_fars,
                            optimizer=optimizer,
                            num_joint_iters=num_joint_iters_,
                        )
                
                torch.cuda.empty_cache()
                
                # run diffusion during mapping
                if self.diffusion_num > 3:
                    folder_path = "diffusion_views_fix"
                    diffusion_idx = [i for i in range(self.diffusion_num)]
                    data_dict = self.get_diffusion_items(folder_path=folder_path, image_size=(H, W), device=self.device, diffusion_idx=diffusion_idx)
                    for frame_idx, frame in enumerate(data_dict.keys()):
                        gt_color, gt_depth, c2w, gt_c2w, mask = data_dict[frame]

                        if gt_depth.min() > 10.0:
                            # print("gt_depth",gt_depth.min())
                            continue
                        else:
                            # print("gt_depth",gt_depth.min())
                            other_indices = [f for i, f in enumerate(data_dict.keys()) if i != frame_idx]
                            num_to_sample = min(3, len(other_indices))
                            sampled = random.sample(other_indices, num_to_sample)
                            frame_indices = [frame] + sampled 
                            visit_rays_o, visit_rays_d, visit_gt_depth, visit_gt_color, volume_feature, uncertainty_map, mvs_depth, outputs, imgs, pose_ref, near_far, near_fars = \
                                self.get_ucnerf_rays(frame_indices, data_dict, H, W, fx, fy, cx, cy)
                            frame_indices = []

                            if len(visit_rays_o) < 100:
                                continue

                            if self.diffusion_views:
                                num_joint_iters_ = 1
                            else:
                                num_joint_iters_ = 1
                            self.optimize_map(
                                rays_o=visit_rays_o,
                                rays_d=visit_rays_d,
                                rays_color=visit_gt_color,
                                rays_depth=visit_gt_depth,
                                volume_feature=volume_feature,
                                uncertainty_map=uncertainty_map,
                                mvs_depth=mvs_depth,
                                outputs=outputs,
                                imgs_input=imgs[:, 1:],
                                pose_ref=pose_ref,
                                near_far=near_far,
                                near_fars=near_fars,
                                optimizer=optimizer,
                                num_joint_iters=num_joint_iters_,
                            )
                torch.cuda.empty_cache()

                print("reload map mapping visit",self.reload_map)
                if self.reload_map > 1000 and the_end:
                    while(self.reload_map > 1000 and the_end):
                        sleep(1.0)
                    self.diffusion_views = True
                    if self.diffusion_views:
                        mesh_root = f'{self.output}/mesh/'
                        extrinsic_matrix = culculate_extrinsic(mesh_root, timestamp, len(visit_list))
                        if extrinsic_matrix is not None:
                            extrinsic_matrix = culculate_extrinsic(mesh_root, timestamp, len(visit_list)+1)
                            extrinsic_matrix = culculate_extrinsic(mesh_root, timestamp, len(visit_list)+2)
                            folder_path = "diffusion_views"
                            diffusion_idx = [str(len(visit_list)-1), str(len(visit_list)), str(len(visit_list)+1), str(len(visit_list)+2)]
                            self.diffusion_num = run_diffusion(diffusion_idx, self.diffusion_num)
                            data_dict = self.get_diffusion_items(folder_path=folder_path, image_size=(H, W), device=self.device, diffusion_idx=diffusion_idx)
                            for frame_idx, frame in enumerate(data_dict.keys()):
                                gt_color, gt_depth, c2w, gt_c2w, mask = data_dict[frame]
            
                                if gt_depth.min() > 10.0:
                                    # print("gt_depth",gt_depth.min())
                                    continue
                                else:
                                    # print("gt_depth",gt_depth.min())
                                    other_indices = [f for i, f in enumerate(data_dict.keys()) if i != frame_idx]
                                    num_to_sample = min(3, len(other_indices))
                                    sampled = random.sample(other_indices, num_to_sample)
                                    frame_indices = [frame] + sampled 
                                    visit_rays_o, visit_rays_d, visit_gt_depth, visit_gt_color, volume_feature, uncertainty_map, mvs_depth, outputs, imgs, pose_ref, near_far, near_fars = \
                                        self.get_ucnerf_rays(frame_indices, data_dict, H, W, fx, fy, cx, cy)
                                    frame_indices = []

                                    if len(visit_rays_o) < 100:
                                        continue

                                    if self.diffusion_views:
                                        num_joint_iters_ = 1
                                    else:
                                        num_joint_iters_ = 1
                                    self.optimize_map(
                                        rays_o=visit_rays_o,
                                        rays_d=visit_rays_d,
                                        rays_color=visit_gt_color,
                                        rays_depth=visit_gt_depth,
                                        volume_feature=volume_feature,
                                        uncertainty_map=uncertainty_map,
                                        mvs_depth=mvs_depth,
                                        outputs=outputs,
                                        imgs_input=imgs[:, 1:],
                                        pose_ref=pose_ref,
                                        near_far=near_far,
                                        near_fars=near_fars,
                                        optimizer=optimizer,
                                        num_joint_iters=num_joint_iters_,
                                    )
                    shutil.rmtree("diffusion_views")
                
                

                # if self.diffusion_views:
                #     # 计算直线的终点（假设长度为0.25）
                #     end_points = visit_rays_o + 0.25 * visit_rays_d
                #     # 创建3D图
                #     fig = plt.figure()
                #     ax = fig.add_subplot(111, projection='3d')
                #     # 绘制箭头
                #     for i in range(len(visit_rays_o)):
                #         # 起点是 unvisit_rays_o[i]
                #         # 方向是 unvisit_rays_d[i]
                #         # 长度为 0.25
                #         ax.quiver(
                #             visit_rays_o[i, 0].cpu().numpy(), visit_rays_o[i, 1].cpu().numpy(), visit_rays_o[i, 2].cpu().numpy(),  # 起点坐标
                #             visit_rays_d[i, 0].cpu().numpy(), visit_rays_d[i, 1].cpu().numpy(), visit_rays_d[i, 2].cpu().numpy(),  # 方向向量
                #             length=0.1,  # 箭头长度
                #             color='b', alpha=0.5, arrow_length_ratio=0.2  # 箭头的长度比例（箭头头部长度与总长度的比例）
                #         )
                #     # 设置坐标轴标签
                #     ax.set_xlabel('X')
                #     ax.set_ylabel('Y')
                #     ax.set_zlabel('Z')
                #     # 保存图片
                #     plt.savefig(f'{self.output}/logs/mapping/output_3d_arrows.png', dpi=300)  # 指定保存路径和文件名，可以调整dpi来改变图片质量
                #     # 关闭图形以释放资源
                #     plt.close(fig)

                

            # 3d mesh has been updated, info the mesher to regenerate mesh
            
            if the_end and iter == 9:
                self.reload_map += (1000-self.reload_map)
            else:
                self.reload_map += 1
            self.init = False
            torch.cuda.empty_cache()
            del visit_frame, unvisit_frame

