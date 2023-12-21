import logging
import os
from collections import namedtuple
from itertools import accumulate
from typing import List, Optional, Union

import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from scipy import ndimage
from tqdm import tqdm
import json

from datasets import SceneDataset
from datasets.utils import voxel_coords_to_world_coords, world_coords_to_voxel_coords
from radiance_fields import DensityField, RadianceField
from radiance_fields.render_utils import render_rays
from third_party.nerfacc_prop_net import PropNetEstimator
from utils.misc import get_robust_pca
from utils.misc import NumpyEncoder

import plyfile
DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

logger = logging.getLogger()
turbo_cmap = cm.get_cmap("turbo")

from sklearn.cluster import DBSCAN
    
camera_list_global = ['camera_SIDE_LEFT', 'camera_FRONT_LEFT', 'camera_FRONT', 'camera_FRONT_RIGHT', 'camera_SIDE_RIGHT']
    
def export_pcl_ply(pcl: np.ndarray, pcl_color: np.ndarray = None, filepath: str = ...):
    """
    pcl_color: if provided, should be uint8_t
    """
    num_pts = pcl.shape[0]
    if pcl_color is not None:
        verts_tuple = np.zeros((num_pts,), dtype=[(
            "x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data = [tuple(p1.tolist() + p2.tolist()) for p1, p2 in zip(pcl, pcl_color)]
        verts_tuple[:] = data[:]
    else:
        verts_tuple = np.zeros((num_pts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        data = [tuple(p.tolist()) for p in pcl]
        verts_tuple[:] = data[:]

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    ply_data = plyfile.PlyData([el_verts])
    print(f"=> Saving pointclouds to {str(filepath)}")
    ply_data.write(filepath)

def to_img_semantic(tensor: torch.Tensor, is_gt=False, is_one_hot=False):
    num_class = 200
    if not is_gt:
        HW, num_class = tensor.shape
    else:
        HW = tensor.shape
    color_mapping_200 = [(174, 53, 103), (51, 197, 5), (125, 174, 79), (216, 180, 33), (40, 98, 236), 
                    (182, 76, 61), (101, 146, 234), (102, 160, 104), (36, 113, 82), (246, 20, 140), 
                    (102, 25, 84), (141, 39, 194), (241, 229, 136), (178, 141, 186), (40, 209, 242), 
                    (187, 255, 192), (70, 156, 181), (11, 198, 120), (241, 79, 33), (232, 98, 18), 
                    (126, 89, 27), (19, 170, 172), (49, 141, 241), (152, 111, 19), (77, 110, 57), 
                    (159, 117, 73), (119, 18, 22), (52, 194, 23), (218, 42, 229), (11, 176, 137), 
                    (95, 43, 144), (98, 102, 210), (145, 209, 108), (12, 139, 57), (122, 239, 209), 
                    (234, 50, 129), (27, 53, 164), (157, 214, 37), (168, 17, 10), (22, 178, 74), 
                    (242, 134, 12), (66, 22, 129), (194, 161, 16), (201, 44, 87), (235, 189, 38), 
                    (202, 59, 143), (195, 112, 34), (23, 76, 132), (231, 57, 209), (86, 114, 161), 
                    (57, 101, 63), (48, 231, 235), (106, 86, 187), (126, 16, 190), (228, 236, 247), 
                    (120, 120, 120), (29, 85, 95), (60, 67, 148), (108, 42, 218), (0, 236, 104), 
                    (238, 134, 73), (243, 165, 184), (139, 58, 51), (75, 1, 91), (148, 52, 4), 
                    (156, 31, 161), (75, 113, 237), (7, 162, 124), (63, 180, 15), (187, 184, 240), 
                    (59, 94, 28), (23, 19, 78), (245, 221, 35), (92, 239, 133), (60, 173, 222), 
                    (197, 138, 41), (129, 238, 41), (148, 114, 97), (204, 13, 57), (78, 162, 220), 
                    (81, 63, 21), (0, 214, 187), (241, 133, 95), (125, 3, 229), (212, 175, 69), 
                    (15, 104, 143), (205, 241, 161), (51, 233, 130), (24, 140, 124), (150, 151, 78), 
                    (92, 80, 221), (195, 155, 193), (2, 17, 244), (109, 5, 148), (34, 30, 214), 
                    (158, 219, 82), (91, 189, 87), (113, 174, 39), (124, 222, 221), (179, 52, 181), 
                    (207, 109, 127), (157, 43, 5), (197, 69, 214), (205, 92, 167), (9, 66, 102), 
                    (128, 2, 243), (234, 36, 40), (137, 129, 255), (119, 98, 151), (89, 157, 17), 
                    (121, 39, 92), (103, 174, 214), (229, 216, 63), (162, 23, 127), (29, 92, 66), 
                    (40, 98, 27), (56, 239, 218), (141, 184, 212), (214, 139, 210), (136, 228, 76), 
                    (105, 21, 18), (92, 195, 206), (68, 25, 238), (130, 175, 194), (82, 188, 224), 
                    (70, 184, 25), (97, 107, 40), (195, 124, 51), (206, 165, 223), (125, 11, 199), 
                    (128, 179, 216), (16, 75, 25), (68, 26, 191), (136, 165, 175), (24, 179, 154), 
                    (171, 126, 204), (167, 30, 234), (197, 230, 99), (3, 134, 207), (207, 89, 239), 
                    (224, 38, 51), (6, 42, 150), (154, 123, 158), (219, 161, 173), (202, 33, 72), 
                    (30, 55, 131), (200, 200, 200), (112, 115, 107), (149, 236, 53), (115, 50, 245), 
                    (109, 15, 209), (21, 76, 175), (169, 193, 197), (114, 190, 132), (134, 21, 11), 
                    (227, 12, 231), (78, 227, 192), (80, 32, 115), (99, 87, 62), (46, 156, 93), 
                    (102, 97, 148), (72, 190, 78), (214, 170, 243), (251, 159, 28), (107, 160, 163), 
                    (3, 19, 42), (61, 105, 85), (219, 50, 217), (175, 105, 38), (255, 86, 175), 
                    (207, 232, 59), (130, 175, 140), (64, 125, 127), (236, 64, 24), (89, 224, 160), 
                    (19, 70, 31), (219, 1, 17), (161, 193, 120), (35, 15, 109), (180, 245, 97), 
                    (153, 247, 106), (19, 101, 148), (194, 150, 11), (164, 245, 238), (209, 231, 57), 
                    (144, 119, 235), (214, 104, 152), (228, 221, 17), (48, 129, 194), (43, 21, 143), 
                    (17, 117, 65), (87, 160, 220), (109, 25, 75), (137, 23, 225), (102, 52, 175), 
                    (168, 84, 67), (237, 47, 218), (48, 173, 36), (238, 100, 199), (139, 230, 243), 
                    (29, 124, 148), (99, 136, 201), (24, 12, 191), (147, 110, 94), (148, 131, 69)]
    color_mapping = color_mapping_200[:num_class]
    color_mapping = torch.tensor(color_mapping, dtype=torch.float, device=tensor.device).reshape(-1,3)/255.0
    
    
    if is_gt:
        class_i = tensor.long()
        one_hot_tensor = torch.nn.functional.one_hot(class_i, num_class).float()
    else:
        if is_one_hot:
            softmax_tensor = tensor
        else:
            softmax_tensor = torch.relu(torch.sign(tensor-0.5))
            # let the wrong cases to background class
            bg_bias = torch.cat([torch.zeros_like(softmax_tensor[..., :1]),softmax_tensor[..., :1].tile(1, num_class-1)], -1)
            softmax_tensor = torch.relu(softmax_tensor - bg_bias)
        
        class_i = torch.argmax(softmax_tensor, dim=-1).long()
        one_hot_tensor = torch.nn.functional.one_hot(class_i, num_class).float()
    rgb_image = one_hot_tensor@color_mapping
    return rgb_image, class_i.reshape([HW, 1]) #.reshape([-1, 3]).data.cpu().movedim(-1,0) #tensor.reshape([cam.intr.H, cam.intr.W, -1]).data.cpu().movedim(-1,0) # [C,H,W]

def remove_selected_points(pts_pred: torch.Tensor, class_name=0, is_one_hot=False, num_class_=200):
    # pts: [N,60]
    if is_one_hot:
        softmax_tensor = pts_pred
    else:
        softmax_tensor = torch.relu(torch.sign(pts_pred-0.5))
        # let the wrong cases to background class
        bg_bias = torch.cat([torch.zeros_like(softmax_tensor[..., :1]),softmax_tensor[..., :1].tile(1, num_class_-1)], -1)
        softmax_tensor = torch.relu(softmax_tensor - bg_bias)
    one_num = torch.argmax(softmax_tensor, dim=-1).long() # [N,60]
    # remove the points of the selected class
    valid = one_num != class_name
    return valid

def keep_selected_points(pts_pred: torch.Tensor, class_name=0, is_one_hot=False, num_class_=200):
    # pts: [N,60]
    if is_one_hot:
        softmax_tensor = pts_pred
    else:
        softmax_tensor = torch.relu(torch.sign(pts_pred-0.5))
        # let the wrong cases to background class
        bg_bias = torch.cat([torch.zeros_like(softmax_tensor[..., :1]),softmax_tensor[..., :1].tile(1, num_class_-1)], -1)
        softmax_tensor = torch.relu(softmax_tensor - bg_bias)
    one_num = torch.argmax(softmax_tensor, dim=-1).long() # [N,60]
    # remove the points of the selected class
    valid = one_num == class_name
    return valid
   
def save_npz_xyzirgb(pts: torch.Tensor, pts_color: torch.Tensor, pts_color_i: torch.Tensor, vid_root, name='xyzirgb.npz'):
    output_file_path_npz = os.path.join(vid_root, name)
    npz_data = torch.cat([pts, pts_color_i, pts_color], dim=-1).cpu()
    np.savez_compressed(output_file_path_npz, np.array(npz_data)) # 
    
def z_score_selection_for_pointcloud(pts: torch.Tensor, pts_color: torch.Tensor, thre=torch.tensor([1.0,1.5,1.0], device='cuda'), class_num=200, repeat=1):
    # Remove outliers
    # pts: [N,3]
    for j in range(repeat):
        new_res_pts = []
        new_res_pts_color = []
        for i in range(1, class_num):
            valid = keep_selected_points(pts_color, i)
            pts_cur_class = pts[valid]
            pts_color_cur_class = pts_color[valid]
            mean = pts_cur_class.mean(dim=0, keepdim=True)
            std = pts_cur_class.std(dim=0, keepdim=True)
            # valid = (pts - mean).norm(dim=-1) < thre * std
            z_score = (pts_cur_class - mean) / std
            outlier_indices_x = z_score.abs() < thre[0]
            outlier_indices_y = z_score.abs() < thre[1]
            outlier_indices_z = z_score.abs() < thre[2]
            outlier_indices = outlier_indices_x[:,0] * outlier_indices_y[:,1] * outlier_indices_z[:,2] 
            # debug
            # cor = ['x', 'y', 'z']
            # for j in range(3):
            #     # Create a histogram of the data
            #     plt.hist(z_score[:,j].abs().cpu().numpy(), bins=100)

            #     # Add labels and title
            #     plt.xlabel('Value')
            #     plt.ylabel('Frequency')
            #     plt.title('Data Distribution')

            #     # Save the plot as an image as name x, y, z
            #     plt.savefig('./'+cor[j]+'/class'+str(i)+'_'+cor[j]+'.png')

            #     plt.close()
            
            new_res_pts.append(pts_cur_class[outlier_indices])
            new_res_pts_color.append(pts_color_cur_class[outlier_indices])
        all_pts = torch.cat(new_res_pts, dim=0)
        all_pts_color = torch.cat(new_res_pts_color, dim=0)
        pts = all_pts
        pts_color = all_pts_color
    rgb, calss_i = to_img_semantic(all_pts_color)
    
    return all_pts, rgb, calss_i, all_pts_color

def DBSCAN_selection_for_pointcloud(pts: torch.Tensor, pts_color: torch.Tensor, class_num=200, min_s=[5, 5], rad=[0.2, 0.2], thre_num=500):
    # Remove outliers
    # pts: [N,3]
    new_res_pts = []
    new_res_pts_color = []
    for i in range(1, class_num):
        valid = keep_selected_points(pts_color, i)
        pts_cur_class = pts[valid]
        if pts_cur_class.shape[0] == 0:
            continue
        pts_color_cur_class = pts_color[valid]
        # 创建DBSCAN对象并设置参数
        if pts_cur_class.shape[0]>thre_num: # TODO: depth
            dbscan = DBSCAN(eps=rad[1], min_samples=min_s[1])
        else:
            dbscan = DBSCAN(eps=rad[0], min_samples=min_s[0])
        # 执行DBSCAN聚类算法
        labels = dbscan.fit_predict(pts_cur_class.cpu()) #.to(pts.device)

        # 获取异常点的索引
        outlier_indices = labels != -1
        # filtered_point_cloud = np.delete(pts_cur_class, outlier_indices, axis=0)
        
        new_res_pts.append(pts_cur_class[outlier_indices])
        new_res_pts_color.append(pts_color_cur_class[outlier_indices])
    all_pts = torch.cat(new_res_pts, dim=0)
    all_pts_color = torch.cat(new_res_pts_color, dim=0)
    rgb, calss_i = to_img_semantic(all_pts_color)
    return all_pts, rgb, calss_i

def check_sphere_intersection(center1, radius1, center2, radius2, margin=0.4):
    # 计算两个球体之间的距离
    distance = ((center1 - center2)**2).sum().sqrt()
    
    # 判断两个球体是否相交
    if distance <= radius1 + radius2 - margin:
        return True
    else:
        return False
    
def merge_pcl(merged_pcl, merged_pcl_color, new_pcl, new_pcl_color, is_one_hot,margin=5):
    # merged_pcl: [N,3]
    # merged_pcl_color: [N,60]
    # new_pcl: [N,3]
    # new_pcl_color: [N,60]
    '''
    1. calculate the center and raduis of the merged_pcl, per class.
    2. calculate the center and raduis of the new_pcl, per class.
    3. check if the two spheres intersect, merge the two class if they intersect.
    4. append the new class of the new_pcl, which not intersect with the merged_pcl.
    '''
    
    # calculate the center and raduis of the merged_pcl, per class. NOTE: the class between merged_pcl and new_pcl is not the same.
    merged_class_cr = []
    new_class_cr = []
    for i in range(1, num_class):
        valid = keep_selected_points(merged_pcl_color, i, is_one_hot=is_one_hot, num_class_=total_class if is_one_hot else num_class)
        merged_pcl_cur_class = merged_pcl[valid]
        if merged_pcl_cur_class.shape[0] != 0:
            merged_pcl_color_cur_class = merged_pcl_color[valid]
            # calc the center cordination of the merged_pcl
            center_ = merged_pcl_cur_class.mean(dim=0)
            # calc the raduis of the merged_pcl
            raduis_ = ((merged_pcl_cur_class - center_)**2).sum(dim=1).sqrt().max().view(1)
            merged_class_cr.append(torch.cat([center_, raduis_, torch.tensor([i], device=center_.device)]))
        
        # repeat the above process for the new_pcl
        valid = keep_selected_points(new_pcl_color, i)
        new_pcl_cur_class = new_pcl[valid]
        if new_pcl_cur_class.shape[0] != 0:
            new_pcl_color_cur_class = new_pcl_color[valid]
            # calc the center cordination of the new_pcl
            center_ = new_pcl_cur_class.mean(dim=0)
            # calc the raduis of the new_pcl
            raduis_ = ((new_pcl_cur_class - center_)**2).sum(dim=1).sqrt().max().view(1)
            new_class_cr.append(torch.cat([center_, raduis_, torch.tensor([i], device=center_.device)]))
            
    # remake the label class
    # valid_0 = keep_selected_points(merged_pcl_color, 0, is_one_hot)
    remerged_pcl = [] #merged_pcl[valid_0]
    remerged_pcl_color = [] #merged_pcl_color[valid]
    change_class_i_m = []
    change_class_i_n = []
    cur_class_i = 1
    # add the merged class and remove the part of the origin
    for i in range(len(merged_class_cr)):
        for j in range(len(new_class_cr)):
            # check if the two spheres intersect
            if check_sphere_intersection(merged_class_cr[i][:3], merged_class_cr[i][3], new_class_cr[j][:3], new_class_cr[j][3], margin=margin):
                # if the two spheres intersect, then merge the two spheres
                one_hot_tensor = torch.nn.functional.one_hot(torch.tensor([cur_class_i], device=merged_pcl.device).long(), total_class).float()
                valid_i_m = keep_selected_points(merged_pcl_color, merged_class_cr[i][4].cpu().item(), is_one_hot=is_one_hot, num_class_=total_class if is_one_hot else num_class)
                valid_i_n = keep_selected_points(new_pcl_color, new_class_cr[j][4].cpu().item())
                cur_merged_tmp_pcl = torch.cat([merged_pcl[valid_i_m], new_pcl[valid_i_n]], dim=0)
                
                remerged_pcl.append(cur_merged_tmp_pcl)
                remerged_pcl_color.append(one_hot_tensor.view(1,-1).tile(cur_merged_tmp_pcl.shape[0],1))
                cur_class_i+=1
                
                # remove the merged class
                remove_valid_i_m = remove_selected_points(merged_pcl_color, merged_class_cr[i][4].cpu().item(), is_one_hot=is_one_hot, num_class_=total_class if is_one_hot else num_class)
                merged_pcl_color = merged_pcl_color[remove_valid_i_m]
                merged_pcl = merged_pcl[remove_valid_i_m]
                remove_valid_i_n = remove_selected_points(new_pcl_color, new_class_cr[j][4].cpu().item())
                new_pcl_color = new_pcl_color[remove_valid_i_n]
                new_pcl = new_pcl[remove_valid_i_n]
                
                # record the class i
                change_class_i_m.append(merged_class_cr[i][4].cpu().item())
                change_class_i_n.append(new_class_cr[j][4].cpu().item())
                break
                
    # append the rest class of the merged_pcl, which not intersect with the new_pcl.
    for i in range(len(merged_class_cr)):
        if merged_class_cr[i][4].cpu().item() not in change_class_i_m:
            one_hot_tensor = torch.nn.functional.one_hot(torch.tensor([cur_class_i], device=merged_pcl.device).long(), total_class).float()
            valid_i_m = keep_selected_points(merged_pcl_color, merged_class_cr[i][4].cpu().item(), is_one_hot=is_one_hot, num_class_=total_class if is_one_hot else num_class)
            remerged_pcl.append(merged_pcl[valid_i_m])
            remerged_pcl_color.append(one_hot_tensor.view(1,-1).tile(merged_pcl[valid_i_m].shape[0],1))
            cur_class_i+=1
    # append the rest class of the new_pcl, which not intersect with the merged_pcl.
    for i in range(len(new_class_cr)):
        if new_class_cr[i][4].cpu().item() not in change_class_i_n:
            one_hot_tensor = torch.nn.functional.one_hot(torch.tensor([cur_class_i], device=merged_pcl.device).long(), total_class).float()
            valid_i_n = keep_selected_points(new_pcl_color, new_class_cr[i][4].cpu().item())
            remerged_pcl.append(new_pcl[valid_i_n])
            remerged_pcl_color.append(one_hot_tensor.view(1,-1).tile(new_pcl[valid_i_n].shape[0],1))
            cur_class_i+=1
    return torch.cat(remerged_pcl, dim=0), torch.cat(remerged_pcl_color, dim=0)

def to8b(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def resize_five_views(imgs: np.array):
    if len(imgs) != 5:
        return imgs
    for idx in [0, -1]:
        img = imgs[idx]
        new_shape = [int(img.shape[1] * 0.46), img.shape[1], 3]
        new_img = np.zeros_like(img)
        new_img[-new_shape[0] :, : new_shape[1], :] = ndimage.zoom(
            img, [new_shape[0] / img.shape[0], new_shape[1] / img.shape[1], 1]
        )
        # clip the image to 0-1
        new_img = np.clip(new_img, 0, 1)
        imgs[idx] = new_img
    return imgs

def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :],
    )
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.
    from mipnerf

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    if lo is None or hi is None:
        lo_auto, hi_auto = weighted_percentile(
            value, weight, [50 - percentile / 2, 50 + percentile / 2]
        )
        # If `lo` or `hi` are None, use the automatically-computed bounds above.
        eps = np.finfo(np.float32).eps
        lo = lo or (lo_auto - eps)
        hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
        )
    if weight is not None:
        value *= weight
    else:
        weight = np.ones_like(value)
    if colormap:
        colorized = colormap(value)[..., :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_depth(
    x, acc=None, lo=None, hi=None, depth_curve_fn=lambda x: -np.log(x + 1e-6)
):
    """Visualizes depth maps."""
    return visualize_cmap(
        x,
        acc,
        cm.get_cmap("turbo"),
        curve_fn=depth_curve_fn,
        lo=lo,
        hi=hi,
        matte_background=False,
    )


def _make_colorwheel(transitions: tuple = DEFAULT_TRANSITIONS) -> torch.Tensor:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    return torch.FloatTensor(colorwheel)


WHEEL = _make_colorwheel()
N_COLS = len(WHEEL)
WHEEL = torch.vstack((WHEEL, WHEEL[0]))  # Make the wheel cyclic for interpolation


def scene_flow_to_rgb(
    flow: torch.Tensor,
    flow_max_radius: Optional[float] = None,
    background: Optional[str] = "dark",
) -> torch.Tensor:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Adapted from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/visualize.py
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )

    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = torch.abs(complex_flow), torch.angle(complex_flow)
    if flow_max_radius is None:
        # flow_max_radius = torch.max(radius)
        flow_max_radius = torch.quantile(radius, 0.99)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((N_COLS - 1) / (2 * np.pi))

    # Interpolate the hues
    angle_fractional, angle_floor, angle_ceil = (
        torch.fmod(angle, 1),
        angle.trunc(),
        torch.ceil(angle),
    )
    angle_fractional = angle_fractional.unsqueeze(-1)
    wheel = WHEEL.to(angle_floor.device)
    float_hue = (
        wheel[angle_floor.long()] * (1 - angle_fractional)
        + wheel[angle_ceil.long()] * angle_fractional
    )
    ColorizationArgs = namedtuple(
        "ColorizationArgs",
        ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"],
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * factors.unsqueeze(-1)

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - factors.unsqueeze(-1) * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, torch.FloatTensor([255, 255, 255])
        )
    else:
        parameters = ColorizationArgs(
            move_hue_on_S_axis, move_hue_on_V_axis, torch.zeros(3)
        )
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    return colors / 255.0


def vis_occ_plotly(
    vis_aabb: List[Union[int, float]],
    coords: np.array = None,
    colors: np.array = None,
    dynamic_coords: List[np.array] = None,
    dynamic_colors: List[np.array] = None,
    x_ratio: float = 1.0,
    y_ratio: float = 1.0,
    z_ratio: float = 0.125,
    size: int = 5,
    black_bg: bool = False,
    title: str = None,
) -> go.Figure:  # type: ignore
    fig = go.Figure()  # start with an empty figure

    if coords is not None:
        # Add static trace
        static_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=colors,
                symbol="square",
            ),
        )
        fig.add_trace(static_trace)

    # Add temporal traces
    if dynamic_coords is not None:
        for i in range(len(dynamic_coords)):
            fig.add_trace(
                go.Scatter3d(
                    x=dynamic_coords[i][:, 0],
                    y=dynamic_coords[i][:, 1],
                    z=dynamic_coords[i][:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=dynamic_colors[i],
                        symbol="diamond",
                    ),
                )
            )
        steps = []
        if coords is not None:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords) + 1),
                    ],  # Include the static trace
                    label=f"Second {i}",
                )
                step["args"][1][0] = True  # Make the static trace always visible
                step["args"][1][i + 1] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)
        else:
            for i in range(len(dynamic_coords)):
                step = dict(
                    method="restyle",
                    args=[
                        "visible",
                        [False] * (len(dynamic_coords)),
                    ],
                    label=f"Second {i}",
                )
                step["args"][1][i] = True  # Toggle i'th temporal trace to "visible"
                steps.append(step)

        sliders = [
            dict(
                active=0,
                pad={"t": 1},
                steps=steps,
                font=dict(color="white") if black_bg else {},  # Update for font color
            )
        ]
        fig.update_layout(sliders=sliders)
    title_font_color = "white" if black_bg else "black"
    if not black_bg:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    else:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="x",
                    showspikes=False,
                    range=[vis_aabb[0], vis_aabb[3]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                yaxis=dict(
                    title="y",
                    showspikes=False,
                    range=[vis_aabb[1], vis_aabb[4]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                zaxis=dict(
                    title="z",
                    showspikes=False,
                    range=[vis_aabb[2], vis_aabb[5]],
                    backgroundcolor="rgb(0, 0, 0)",
                    gridcolor="gray",
                    showbackground=True,
                    zerolinecolor="gray",
                    tickfont=dict(color="gray"),
                ),
                aspectmode="manual",
                aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
            ),
            margin=dict(r=0, b=10, l=0, t=10),
            hovermode=False,
            paper_bgcolor="black",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text=title,
                font=dict(color=title_font_color),
                x=0.5,
                y=0.95,
                xanchor="center",
                yanchor="top",
            )
            if title
            else None,  # Title addition
        )
    eye = np.array([-1, 0, 0.5])
    eye = eye.tolist()
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        ),
    )
    return fig


def visualize_voxels(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_estimator: PropNetEstimator = None,
    proposal_networks: DensityField = None,
    dataset: SceneDataset = None,
    device: str = "cuda",
    save_html: bool = True,
    is_dynamic: bool = False,
):
    model.eval()
    for p in proposal_networks:
        p.eval()
    if proposal_estimator is not None:
        proposal_estimator.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()

    vis_voxel_aabb = torch.tensor(model.aabb, device=device)
    # slightly expand the aabb to make sure all points are covered
    vis_voxel_aabb[1:3] -= 1
    vis_voxel_aabb[3:] += 1
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    aabb_length = aabb_max - aabb_min

    # compute the voxel resolution for visualization
    static_voxel_resolution = torch.ceil(
        (aabb_max - aabb_min) / cfg.render.vis_voxel_size
    ).long()
    empty_static_voxels = torch.zeros(*static_voxel_resolution, device=device)
    if is_dynamic:
        # use a slightly smaller voxel size for dynamic voxels
        dynamic_voxel_resolution = torch.ceil(
            (aabb_max - aabb_min) / cfg.render.vis_voxel_size * 0.8
        ).long()
        all_occupied_dynamic_points = []
        empty_dynamic_voxels = torch.zeros(*dynamic_voxel_resolution, device=device)

    # collect some patches for PCA
    to_compute_pca_patches = []

    pbar = tqdm(
        dataset.full_pixel_set,
        desc="querying depth",
        dynamic_ncols=True,
        total=len(dataset.full_pixel_set),
    )
    for i, data_dict in enumerate(pbar):
        data_dict = dataset.full_pixel_set[i]
        for k, v in data_dict.items():
            data_dict[k] = v.to(device)
        if i < dataset.num_cams:
            # collect all patches from the first timestep
            with torch.no_grad():
                render_results = render_rays(
                    radiance_field=model,
                    proposal_estimator=proposal_estimator,
                    proposal_networks=proposal_networks,
                    data_dict=data_dict,
                    cfg=cfg,
                    proposal_requires_grad=False,
                )
            if "dino_pe_free" in render_results:
                dino_feats = render_results["dino_pe_free"]
            else:
                dino_feats = render_results["dino_feat"]
            dino_feats = dino_feats.reshape(-1, dino_feats.shape[-1])
            to_compute_pca_patches.append(dino_feats)
        # query the depth. we force a lidar mode here so that the renderer will skip
        # querying other features such as colors, features, etc.
        data_dict["lidar_origins"] = data_dict["origins"].to(device)
        data_dict["lidar_viewdirs"] = data_dict["viewdirs"].to(device)
        data_dict["lidar_normed_timestamps"] = data_dict["normed_timestamps"].to(device)
        with torch.no_grad():
            render_results = render_rays(
                radiance_field=model,
                proposal_estimator=proposal_estimator,
                proposal_networks=proposal_networks,
                data_dict=data_dict,
                cfg=cfg,
                proposal_requires_grad=False,
                prefix="lidar_",  # force lidar mode
                return_decomposition=True,
            )
        # ==== get the static voxels ======
        if is_dynamic:
            static_depth = render_results["static_depth"]
        else:
            static_depth = render_results["depth"]
        world_coords = (
            data_dict["lidar_origins"] + data_dict["lidar_viewdirs"] * static_depth
        )
        world_coords = world_coords[static_depth.squeeze() < 80]
        voxel_coords = world_coords_to_voxel_coords(
            world_coords, aabb_min, aabb_max, static_voxel_resolution
        )
        voxel_coords = voxel_coords.long()
        selector = (
            (voxel_coords[..., 0] >= 0)
            & (voxel_coords[..., 0] < static_voxel_resolution[0])
            & (voxel_coords[..., 1] >= 0)
            & (voxel_coords[..., 1] < static_voxel_resolution[1])
            & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < static_voxel_resolution[2])
        )
        # split the voxel_coords into separate dimensions
        voxel_coords_x = voxel_coords[..., 0][selector]
        voxel_coords_y = voxel_coords[..., 1][selector]
        voxel_coords_z = voxel_coords[..., 2][selector]
        # index into empty_voxels using the separated coordinates
        empty_static_voxels[voxel_coords_x, voxel_coords_y, voxel_coords_z] = 1

        # ==== get the dynamic voxels ======
        if is_dynamic:
            dynamic_depth = render_results["dynamic_depth"]
            world_coords = (
                data_dict["lidar_origins"] + data_dict["lidar_viewdirs"] * dynamic_depth
            )
            voxel_coords = world_coords_to_voxel_coords(
                world_coords, aabb_min, aabb_max, dynamic_voxel_resolution
            )
            voxel_coords = voxel_coords.long()
            selector = (
                (voxel_coords[..., 0] >= 0)
                & (voxel_coords[..., 0] < dynamic_voxel_resolution[0])
                & (voxel_coords[..., 1] >= 0)
                & (voxel_coords[..., 1] < dynamic_voxel_resolution[1])
                & (voxel_coords[..., 2] >= 0)
                & (voxel_coords[..., 2] < dynamic_voxel_resolution[2])
            )
            # split the voxel_coords into separate dimensions
            voxel_coords_x = voxel_coords[..., 0][selector]
            voxel_coords_y = voxel_coords[..., 1][selector]
            voxel_coords_z = voxel_coords[..., 2][selector]
            # index into empty_voxels using the separated coordinates
            empty_dynamic_voxels[voxel_coords_x, voxel_coords_y, voxel_coords_z] = 1
            if i % dataset.num_cams == 0 and i > 0:
                all_occupied_dynamic_points.append(
                    voxel_coords_to_world_coords(
                        aabb_min,
                        aabb_max,
                        dynamic_voxel_resolution,
                        torch.nonzero(empty_dynamic_voxels),
                    )
                )
                empty_dynamic_voxels = torch.zeros(
                    *dynamic_voxel_resolution, device=device
                )
    # compute the pca reduction
    dummy_pca_reduction, color_min, color_max = get_robust_pca(
        torch.cat(to_compute_pca_patches, dim=0).to(device), m=2.5
    )
    # now let's query the features
    all_occupied_static_points = voxel_coords_to_world_coords(
        aabb_min, aabb_max, static_voxel_resolution, torch.nonzero(empty_static_voxels)
    )
    chunk = 2**18
    pca_colors = []
    occupied_points = []
    pbar = tqdm(
        range(0, all_occupied_static_points.shape[0], chunk),
        desc="querying static features",
        dynamic_ncols=True,
    )
    for i in pbar:
        occupied_points_chunk = all_occupied_static_points[i : i + chunk]
        density_list = []
        # we need to accumulate the density from all proposal networks as well
        # to ensure reliable density estimation
        for p in proposal_networks:
            density_list.append(p(occupied_points_chunk)["density"].squeeze(-1))
        with torch.no_grad():
            results = model.forward(
                occupied_points_chunk,
                query_feature_head=False,
            )
        density_list.append(results["density"])
        density = torch.stack(density_list, dim=0)
        density = torch.mean(density, dim=0)
        # use a preset threshold to determine whether a voxel is occupied
        selector = density > 0.5
        occupied_points_chunk = occupied_points_chunk[selector]
        if len(occupied_points_chunk) == 0:
            # skip if no occupied points in this chunk
            continue
        with torch.no_grad():
            feats = model.forward(
                occupied_points_chunk,
                query_feature_head=True,
                query_pe_head=False,
            )["dino_feat"]
        colors = feats @ dummy_pca_reduction
        del feats
        colors = (colors - color_min) / (color_max - color_min)
        pca_colors.append(torch.clamp(colors, 0, 1))
        occupied_points.append(occupied_points_chunk)

    pca_colors = torch.cat(pca_colors, dim=0)
    occupied_points = torch.cat(occupied_points, dim=0)
    if is_dynamic:
        dynamic_pca_colors = []
        dynamic_occupied_points = []
        unq_timestamps = dataset.pixel_source.unique_normalized_timestamps.to(device)
        # query every 10 frames
        pbar = tqdm(
            range(0, len(all_occupied_dynamic_points), 10),
            desc="querying dynamic fields",
            dynamic_ncols=True,
        )
        for i in pbar:
            occupied_points_chunk = all_occupied_dynamic_points[i]
            normed_timestamps = unq_timestamps[i].repeat(
                occupied_points_chunk.shape[0], 1
            )
            with torch.no_grad():
                results = model.forward(
                    occupied_points_chunk,
                    data_dict={"normed_timestamps": normed_timestamps},
                    query_feature_head=False,
                )
            selector = results["dynamic_density"].squeeze() > 0.1
            occupied_points_chunk = occupied_points_chunk[selector]
            if len(occupied_points_chunk) == 0:
                continue
            # query some features
            normed_timestamps = unq_timestamps[i].repeat(
                occupied_points_chunk.shape[0], 1
            )
            with torch.no_grad():
                feats = model.forward(
                    occupied_points_chunk,
                    data_dict={"normed_timestamps": normed_timestamps},
                    query_feature_head=True,
                    query_pe_head=False,
                )["dynamic_dino_feat"]
            colors = feats @ dummy_pca_reduction
            del feats
            colors = (colors - color_min) / (color_max - color_min)
            dynamic_pca_colors.append(torch.clamp(colors, 0, 1))
            dynamic_occupied_points.append(occupied_points_chunk)
        dynamic_coords = [x.cpu().numpy() for x in dynamic_occupied_points]
        dynamic_colors = [x.cpu().numpy() for x in dynamic_pca_colors]
    else:
        dynamic_coords = None
        dynamic_colors = None

    figure = vis_occ_plotly(
        vis_aabb=vis_voxel_aabb.cpu().numpy().tolist(),
        coords=occupied_points.cpu().numpy(),
        colors=pca_colors.cpu().numpy(),
        dynamic_coords=dynamic_coords,
        dynamic_colors=dynamic_colors,
        x_ratio=1,
        y_ratio=(aabb_length[1] / aabb_length[0]).item(),
        z_ratio=(aabb_length[2] / aabb_length[0]).item(),
        size=3,
        black_bg=True,
        title=f"Lifted {cfg.data.pixel_source.feature_model_type} Features, PE_removed: {cfg.nerf.model.head.enable_learnable_pe}",
    )
    # for plotly
    data = figure.to_dict()["data"]
    layout = figure.to_dict()["layout"]
    output_path = os.path.join(cfg.log_dir, f"feature_field.json")
    with open(output_path, "w") as f:
        json.dump({"data": data, "layout": layout}, f, cls=NumpyEncoder)
    logger.info(f"Saved to {output_path}")
    output_path = os.path.join(cfg.log_dir, f"feature_field.html")
    if save_html:
        figure.write_html(output_path)
        logger.info(f"Query result saved to {output_path}")

def visualize_pointscloud(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_estimator: PropNetEstimator = None,
    proposal_networks: DensityField = None,
    dataset: SceneDataset = None,
    device: str = "cuda",
    save_html: bool = True,
    is_dynamic: bool = False,
    denoise_method = 'DBSCAN',
):
    model.eval()
    for p in proposal_networks:
        p.eval()
    if proposal_estimator is not None:
        proposal_estimator.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()

    vis_voxel_aabb = torch.tensor(model.aabb, device=device)
    # slightly expand the aabb to make sure all points are covered
    vis_voxel_aabb[1:3] -= 1
    vis_voxel_aabb[3:] += 1
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    aabb_length = aabb_max - aabb_min

    # compute the voxel resolution for visualization
    static_voxel_resolution = torch.ceil(
        (aabb_max - aabb_min) / cfg.render.vis_voxel_size
    ).long()
    empty_static_voxels = torch.zeros(*static_voxel_resolution, device=device)
    if is_dynamic:
        # use a slightly smaller voxel size for dynamic voxels
        dynamic_voxel_resolution = torch.ceil(
            (aabb_max - aabb_min) / cfg.render.vis_voxel_size * 0.8
        ).long()
        all_occupied_dynamic_points = []
        empty_dynamic_voxels = torch.zeros(*dynamic_voxel_resolution, device=device)

    dataset.pixel_source.update_downscale_factor(1 / cfg.render.low_res_downscale)
    # collect some patches for PCA
    to_compute_pca_patches = []
    # render per frame
    for j in range(len(dataset.full_pixel_set)//dataset.num_cams):
        start = j*dataset.num_cams
        end = (j+1)*dataset.num_cams
        cam_pcl_ = []
        cam_pcl_color_ = []
        for i in range(start, end):
            data_dict = dataset.full_pixel_set[i]
            for k, v in data_dict.items():
                data_dict[k] = v.to(device)
                # collect all patches from the first timestep
            with torch.no_grad():
                render_results = render_rays(
                    radiance_field=model,
                    proposal_estimator=proposal_estimator,
                    proposal_networks=proposal_networks,
                    data_dict=data_dict,
                    cfg=cfg,
                    proposal_requires_grad=False,
                )
            # calc the world coords according to the depth
            depth = render_results["depth"].cuda()
            world_coords = (
                data_dict["origins"] + data_dict["viewdirs"] * depth
            )
            selected = depth.squeeze() < 80
            world_coords = world_coords[selected].view(-1,3).cpu()
            semantics = render_results["semantics"][selected.cpu()]
            # calc the semantic labels to color
            semantics = render_results["semantics"] # [H, W, N]
            # semantics = semantics.view(-1,semantics.shape[-1])
            semantics = semantics[selected.cpu()]
            posi_selected = remove_selected_points(semantics, 0)
            world_coords = world_coords[posi_selected]
            semantics = semantics[posi_selected]
            if world_coords.shape[0] == 0:
                continue
            cam_pcl_.append(world_coords)
            cam_pcl_color_.append(semantics)
        if len(cam_pcl_) == 0:
            continue
        vid_root = os.path.join(cfg.log_dir, 'pointcloud')
        if not os.path.exists(vid_root):
            os.makedirs(vid_root)
        # use DBSCAN in current frame
        # if denoise_method == 'DBSCAN':
        cam_pcl, cam_pcl_color, cam_pcl_i = DBSCAN_selection_for_pointcloud(torch.cat(cam_pcl_, 0).view(-1,3), torch.cat(cam_pcl_color_, 0), min_s=5, rad=0.2)
        save_npz_xyzirgb(cam_pcl, (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8), cam_pcl_i, vid_root, str(j).zfill(3)+'_xyzirgb_DBSCAN.npz')

        cam_pcl = cam_pcl.data.cpu().numpy()
        cam_pcl_color = (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8).cpu().numpy()
        export_pcl_ply(cam_pcl, cam_pcl_color, filepath=os.path.join(vid_root, str(j).zfill(3)+'_DBSCAN.ply'))
        # use zscore
        # elif denoise_method == 'zscore':
        # cam_pcl, cam_pcl_color, cam_pcl_i, _ = z_score_selection_for_pointcloud(torch.cat(cam_pcl_, 0).view(-1,3), torch.cat(cam_pcl_color_, 0), torch.tensor([1.5,1.5,1.5], device='cpu'))
        # save_npz_xyzirgb(cam_pcl, (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8), cam_pcl_i, vid_root, str(j).zfill(3)+'_xyzirgb_Zscore.npz')

        # cam_pcl = cam_pcl.data.cpu().numpy()
        # cam_pcl_color = (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8).cpu().numpy()
        # export_pcl_ply(cam_pcl, cam_pcl_color, filepath=os.path.join(vid_root, str(j).zfill(3)+'_Zcore.ply'))
        
def visualize_pointscloud_wholeScene(
    cfg: OmegaConf,
    model: RadianceField,
    proposal_estimator: PropNetEstimator = None,
    proposal_networks: DensityField = None,
    dataset: SceneDataset = None,
    device: str = "cuda",
    save_html: bool = True,
    is_dynamic: bool = False,
    denoise_method = 'DBSCAN',
):
    model.eval()
    for p in proposal_networks:
        p.eval()
    if proposal_estimator is not None:
        proposal_estimator.eval()
    if proposal_networks is not None:
        for p in proposal_networks:
            p.eval()

    vis_voxel_aabb = torch.tensor(model.aabb, device=device)
    # slightly expand the aabb to make sure all points are covered
    vis_voxel_aabb[1:3] -= 1
    vis_voxel_aabb[3:] += 1
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    aabb_length = aabb_max - aabb_min

    # compute the voxel resolution for visualization
    static_voxel_resolution = torch.ceil(
        (aabb_max - aabb_min) / cfg.render.vis_voxel_size
    ).long()
    empty_static_voxels = torch.zeros(*static_voxel_resolution, device=device)
    if is_dynamic:
        # use a slightly smaller voxel size for dynamic voxels
        dynamic_voxel_resolution = torch.ceil(
            (aabb_max - aabb_min) / cfg.render.vis_voxel_size * 0.8
        ).long()
        all_occupied_dynamic_points = []
        empty_dynamic_voxels = torch.zeros(*dynamic_voxel_resolution, device=device)

    dataset.pixel_source.update_downscale_factor(1 / cfg.render.low_res_downscale)
    # render per time
    progress_bar = tqdm(range(len(dataset.full_pixel_set)//dataset.num_cams))
    for t in progress_bar:
        current_time = dataset.full_pixel_set[t*5]['normed_timestamps'][0]
        cam_pcl_ = []
        cam_pcl_color_ = []
        progress_bar_curT = tqdm(range(len(dataset.full_pixel_set)//dataset.num_cams))
        for j in progress_bar_curT:
            start = j*dataset.num_cams
            end = (j+1)*dataset.num_cams
            for i in range(start, end):
                data_dict = dataset.full_pixel_set[i]
                data_dict['normed_timestamps'] = torch.ones_like(data_dict['normed_timestamps']) * current_time
                for k, v in data_dict.items():
                    data_dict[k] = v.to(device)
                    # collect all patches from the first timestep
                with torch.no_grad():
                    render_results = render_rays(
                        radiance_field=model,
                        proposal_estimator=proposal_estimator,
                        proposal_networks=proposal_networks,
                        data_dict=data_dict,
                        cfg=cfg,
                        proposal_requires_grad=False,
                    )
                # calc the world coords according to the depth
                depth = render_results["depth"].cuda()
                world_coords = (
                    data_dict["origins"] + data_dict["viewdirs"] * depth
                )
                selected = depth.squeeze() < 80
                world_coords = world_coords[selected].view(-1,3).cpu()
                semantics = render_results["semantics"][selected.cpu()]
                # calc the semantic labels to color
                semantics = render_results["semantics"] # [H, W, N]
                # semantics = semantics.view(-1,semantics.shape[-1])
                semantics = semantics[selected.cpu()]
                posi_selected = remove_selected_points(semantics, 0)
                world_coords = world_coords[posi_selected]
                semantics = semantics[posi_selected]
                if world_coords.shape[0] == 0:
                    continue
                cam_pcl_.append(world_coords)
                cam_pcl_color_.append(semantics)
            progress_bar_curT.set_description(f"Processing {j+1}/{len(dataset.full_pixel_set)//dataset.num_cams} of {t}th frame")
        progress_bar.set_description(f"Processing {j+1}/{len(dataset.full_pixel_set)//dataset.num_cams} of {t}th frame")

        vid_root = os.path.join(cfg.log_dir, 'pointcloud')
        if not os.path.exists(vid_root):
            os.makedirs(vid_root)
        # use DBSCAN in current frame
        # if denoise_method == 'DBSCAN':
        cam_pcl, cam_pcl_color, cam_pcl_i = DBSCAN_selection_for_pointcloud(torch.cat(cam_pcl_, 0).view(-1,3), torch.cat(cam_pcl_color_, 0), min_s=[5, 10], rad=[0.5, 0.2], thre_num=500)
        save_npz_xyzirgb(cam_pcl, (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8), cam_pcl_i, vid_root, str(t).zfill(3)+'_xyzirgb_DBSCAN.npz')

        cam_pcl = cam_pcl.data.cpu().numpy()
        cam_pcl_color = (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8).cpu().numpy()
        export_pcl_ply(cam_pcl, cam_pcl_color, filepath=os.path.join(vid_root, str(t).zfill(3)+'_DBSCAN.ply'))
        # use zscore
        # elif denoise_method == 'zscore':
        # cam_pcl, cam_pcl_color, cam_pcl_i, _ = z_score_selection_for_pointcloud(torch.cat(cam_pcl_, 0).view(-1,3), torch.cat(cam_pcl_color_, 0), torch.tensor([1.5,1.5,1.5], device='cpu'))
        # save_npz_xyzirgb(cam_pcl, (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8), cam_pcl_i, vid_root, str(t).zfill(3)+'_xyzirgb_Zscore.npz')

        # cam_pcl = cam_pcl.data.cpu().numpy()
        # cam_pcl_color = (cam_pcl_color.data*255.).clamp_(0., 255.).to(dtype=torch.uint8).cpu().numpy()
        # export_pcl_ply(cam_pcl, cam_pcl_color, filepath=os.path.join(vid_root, str(t).zfill(3)+'_Zcore.ply'))
        

def visualize_scene_flow(
    cfg: OmegaConf,
    model: RadianceField,
    dataset: SceneDataset = None,
    device: str = "cuda",
    save_html: bool = True,
):
    pbar = tqdm(
        range(0, len(dataset.full_lidar_set) - 1, 10),
        desc="querying flow",
        dynamic_ncols=True,
    )
    predicted_flow_colors, gt_flow_colors = [], []
    dynamic_coords = []
    for i in pbar:
        data_dict = dataset.full_lidar_set[i].copy()
        lidar_flow_class = data_dict["lidar_flow_class"]
        for k, v in data_dict.items():
            # remove invalid flow (the information is from GT)
            data_dict[k] = v[lidar_flow_class != -1]

        if data_dict[k].shape[0] == 0:
            logger.info(f"no valid points, skipping...")
            continue
        # filter out ground points
        # for k, v in data_dict.items():
        #     data_dict[k] = v[~data_dict["lidar_ground"]]
        valid_lidar_mask = dataset.get_valid_lidar_mask(i, data_dict)
        for k, v in data_dict.items():
            data_dict[k] = v[valid_lidar_mask]
        lidar_points = (
            data_dict["lidar_origins"]
            + data_dict["lidar_ranges"] * data_dict["lidar_viewdirs"]
        )
        normalized_timestamps = data_dict["lidar_normed_timestamps"]
        with torch.no_grad():
            pred_results = model.query_flow(
                positions=lidar_points,
                normed_timestamps=normalized_timestamps,
            )
        pred_flow = pred_results["forward_flow"]
        # flow is only valid when the point is not static
        pred_flow[pred_results["dynamic_density"] < 0.2] *= 0

        predicted_flow_colors.append(
            scene_flow_to_rgb(pred_flow, flow_max_radius=2.0, background="bright")
            .cpu()
            .numpy()
        )
        gt_flow_colors.append(
            scene_flow_to_rgb(
                data_dict["lidar_flow"], flow_max_radius=2.0, background="bright"
            )
            .cpu()
            .numpy()
        )
        dynamic_coords.append(lidar_points.cpu().numpy())

    vis_voxel_aabb = torch.tensor(model.aabb, device=device)
    # slightly expand the aabb to make sure all points are covered
    vis_voxel_aabb[1:3] -= 1
    vis_voxel_aabb[3:] += 1
    aabb_min, aabb_max = torch.split(vis_voxel_aabb, 3, dim=-1)
    aabb_length = aabb_max - aabb_min
    pred_figure = vis_occ_plotly(
        vis_aabb=vis_voxel_aabb.cpu().numpy().tolist(),
        dynamic_coords=dynamic_coords,
        dynamic_colors=predicted_flow_colors,
        x_ratio=1,
        y_ratio=(aabb_length[1] / aabb_length[0]).item(),
        z_ratio=(aabb_length[2] / aabb_length[0]).item(),
        size=2,
        black_bg=True,
        title=f"Predicted Flow",
    )
    gt_figure = vis_occ_plotly(
        vis_aabb=vis_voxel_aabb.cpu().numpy().tolist(),
        dynamic_coords=dynamic_coords,
        dynamic_colors=gt_flow_colors,
        x_ratio=1,
        y_ratio=(aabb_length[1] / aabb_length[0]).item(),
        z_ratio=(aabb_length[2] / aabb_length[0]).item(),
        size=2,
        black_bg=True,
        title=f"GT Flow",
    )
    if save_html:
        output_path = os.path.join(cfg.log_dir, f"predicted_flow.html")
        pred_figure.write_html(output_path)
        logger.info(f"Predicted flow result saved to {output_path}")
        output_path = os.path.join(cfg.log_dir, f"gt_flow.html")
        gt_figure.write_html(output_path)
        logger.info(f"GT flow saved to {output_path}")
