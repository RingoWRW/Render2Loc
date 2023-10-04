import torch
import os
import numpy as np
import glob
from hloc.transform import parse_intrinsic_list
from tqdm import tqdm
import json
import pyproj
import math
from scipy.spatial.transform import Rotation as R
import os
import argparse
from pathlib import Path
from  render2loc.lib import (
    localize_render2loc,
    match_feature,
    generate_seed
)
import exifread
from pyexiv2 import Image
from hloc import  ray_casting, ray_casting_east

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default='/media/ubuntu/DB/render2loc',#!
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='/media/ubuntu/DB/render2loc/outputs',#!
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=20,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=1,
                    help='Number of image pairs for loc, default: %(default)s')

args = parser.parse_args()
config = {
    "ray_casting": {
        "object_name": "T-barrier0",
        "num_sample": 100,
        "DSM_path": "/media/ubuntu/DB/DSM/Production_5_DSM_merge.tif",
        "DSM_npy_path": "/media/ubuntu/DB/DSM/DSM_array.npy",
        "area_minZ": 20.580223,
        "write_path": "./predictXYZ.txt"
    },

}

# import mathutils
def max_pool(x, nms_radius: int):
    return torch.nn.functional.max_pool2d(
        x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores, nms_radius)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float(), nms_radius) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def json_read(folder_path, query_intrinsics):
    K_w2c = parse_intrinsic_list(query_intrinsics)
    names = list(K_w2c.keys())
    name_list = []
    pose = []
    Points_3D = []
    point3D = []
    shape = []
    pbar = tqdm(total=len(names), unit='pts')
    per_points_num = []
    with open(folder_path, encoding = 'utf-8') as a:
        result = json.load(a)
        for name in tqdm(names):  
            # read file
            if name.split('.')[0] in result.keys():
                name_list.append(name)
                point3D = result.get(name.split('.')[0]).get("Points3D")
                
                per_points_num.append(len(point3D))
                
                for id, point in point3D.items():
                    Points_3D.append(point['xyz'])
                    shape.append(point['shape'])
                pose.append(result.get(name.split('.')[0]).get("transformation"))
            pbar.update(1)
        pbar.close() 
    return Points_3D, per_points_num, pose, name_list, shape

# TODO: nms, input: point[n, 3], point ID[n, 1], per_img_num[img num, 1]
def cgcs2000towgs84(poses, save_path):
    gpss = []
    num = 0
    
    for c2w_t in poses:
        # pose = np.array(pose)
        # import ipdb; ipdb.set_trace();
        # c2w_t = pose[:3, 3]
        x, y, z = c2w_t[0], c2w_t[1], c2w_t[2]
        wgs84 = pyproj.CRS('EPSG:4326')
        cgcs2000 = pyproj.CRS('EPSG:4548')  #117E
        transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84, always_xy=True)
        lon, lat = transformer.transform(x, y)
        gps = np.array([lon, lat, z])
        gpss.append(gps)
        num += 1
        print(num)
    with open(save_path, 'w') as f:
        for tvec in gpss:
            tvec = ' '.join(map(str, tvec))
            f.write(f'{tvec}\n')
    return gpss
def rotation2euler(poses):
    rot_eulrs = []
    for pose in poses:
        pose = np.array(pose)
        c2w_R = pose[:3, :3]
        c2w_R = np.asmatrix(c2w_R)
        c2w_R = R.from_matrix(c2w_R)
        rot_eulr = c2w_R.as_euler('xyz', degrees=True)   
        rot_eulrs.append(rot_eulr.tolist()) 
    return  rot_eulrs
def nms_my(in_corners,shape,  dist_thresh):
    folder_path = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/output2.json"
    index = []
    json_list = []
    in_corner_first = in_corners.copy()
    # inds = np
    in_corners[0, :] = in_corners[0, :] * 10
    in_corners[1, :] = in_corners[1, :] * 10
    max_w = np.max(in_corners[0, :])
    min_w = np.min(in_corners[0, :])
    max_h = np.max(in_corners[1, :])
    min_h = np.min(in_corners[1, :])
    in_corners[0, :] = in_corners[0, :] -min_w 
    in_corners[1, :] = in_corners[1, :] -min_h
    H = math.ceil((max_h - min_h )/ dist_thresh)
    W = math.ceil((max_w - min_w)/ dist_thresh)  
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points. 
    
    rcorners = in_corners[:2,:].round().astype(int) # Rounded corners. #!
    flag = 0
    for i, rc in enumerate(rcorners.T):
        y = int((in_corners[0, i] ) / dist_thresh)  # get coarse patch index
        x = int((in_corners[1, i] ) / dist_thresh)

        if inds[x, y] != 1:
            object_dict = {}
            inds[x, y] = 1
            grid[x, y] = flag
            index.append(grid[x, y])
            object_dict['object_id'] = flag
            x, y, z = in_corner_first[0, i], in_corner_first[1, i], in_corner_first[2, i]
            wgs84 = pyproj.CRS('EPSG:4326')
            cgcs2000 = pyproj.CRS('EPSG:4548')  #117E
            transformer = pyproj.Transformer.from_crs(cgcs2000, wgs84, always_xy=True)
            lon, lat = transformer.transform(x, y)
            gps = np.array([lon, lat, z])
            object_dict['longitude'] = lon
            object_dict['latitude'] = lat
            object_dict['height'] = z
            object_dict['shape'] = shape[i]
            json_list.append(object_dict)
            print(flag)
            flag += 1
            # import ipdb; ipdb.set_trace();
        else:
            index.append(grid[x, y])
    print("intput numbers: ", len(index))
    print("total object numbers: ", flag - 1)
    
    
    with open(folder_path, 'w', encoding='utf-8') as a:
        json.dump(json_list, a, ensure_ascii = False, indent = 2)
            
    return index
    # import ipdb; ipdb.set_trace();
    
def nms_fast(in_corners,  dist_thresh ):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    in_corners[0, :] = in_corners[0, :] * 10
    in_corners[1, :] = in_corners[1, :] * 10
    max_w = np.max(in_corners[0, :])
    min_w = np.min(in_corners[0, :])
    max_h = np.max(in_corners[1, :])
    min_h = np.min(in_corners[1, :])
    in_corners[0, :] = in_corners[0, :] -min_w 
    in_corners[1, :] = in_corners[1, :] -min_h
    dist_thresh = 1
    H = math.ceil(max_h - min_h) + 10
    W = math.ceil(max_w - min_w) + 10
    # print("===", np.max(in_corners[1, :]), np.max(in_corners[0, :]), H, W )
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    index = {}
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    # corners = in_corners[:,inds1]
    # rcorners = corners[:2,:].round().astype(int)
    corners = in_corners
    rcorners = corners[:2,:].round().astype(int) # Rounded corners. #!
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        # import ipdb; ipdb.set_trace();
        print(rcorners[1, i], rcorners[0,i])
        grid[rcorners[1, i], rcorners[0,i]] = 1
        inds[rcorners[1, i], rcorners[0,i]] = i
        
        
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh 
    # grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    import ipdb; ipdb.set_trace();
    return out, out_inds
def generate_points_list(name_list):
    center_points_dict = {}
    center_points = [{'tl': [0,0]},
                        {'tr': [0, 3040]},
                        {'bl': [4056, 0]},
                        {'br': [4056, 3040]},
                        {'center':[2028, 1520]}]
    for name in name_list:
        center_points_dict[name.split('.')[0]] = center_points
    
    return center_points_dict
def get_pixel_gps(name_list, render_poses, render_camera, results_position):
    center_points_dict = generate_points_list(name_list)
    area_minZ = ray_casting.DSM2npy(config["ray_casting"])
    config["ray_casting"]["area_minZ"] = area_minZ
    ray_casting_east.main_foreast_center(config["ray_casting"], render_poses, render_camera, results_position, center_points_dict)
def read_date(exif_file):
    with open(exif_file, "rb") as f:
            # 读取exif信息
            exif_data = exifread.process_file(f)
            date = exif_data['EXIF DateTimeDigitized'].values  
    return date
def update_point3D_index(folder_path, query_intrinsics, name_list, Points_3D, 
                         index, euler, gps, gps_point,gps_pixel,  per_points_num, shape):
    pic_path = "/home/ubuntu/Documents/code/SensLoc/datasets/East/raw/color_images/"
    pose = []
    pbar = tqdm(total=len(name_list), unit='pts')
    json_list = []
    num = 0
    with open(folder_path, 'w', encoding='utf-8') as a:
        for i in tqdm(range(len(per_points_num))): 
            # print(len(per_points_num))
                # print(per_points_num[i])
            pic_dict = {}
            pic_dict["filename"] = name_list[i]
            pic_dict["longtitude"] = gps[i][0]
            pic_dict["latitude"] = gps[i][1]
            pic_dict["height"] = gps[i][2]
            pic_dict["direction"] = euler[i][2] #!xyz
            name = name_list[i]
            date = read_date(pic_path+name)
            pic_dict["date"] = date #!
            objects = {}     
            img_positions = {}
            img_positions["top_left"] = gps_pixel[5 * i]
            img_positions["top_right"] = gps_pixel[5 * i +1]
            img_positions["bottom_left"] = gps_pixel[5 * i +2]
            img_positions["bottom_right"] = gps_pixel[5 * i + 3]
            img_positions["center"] = gps_pixel[5 * i + 4]
            pic_dict["five_corners"] = img_positions
            pic_dict["gps"] = gps[i]
                   
            for j in range(per_points_num[i]): 
                objects[str(index[num])] = {"lon": gps_point[num][0], 
                                            "lat": gps_point[num][1],
                                            "height": gps_point[num][2],
                                            "type": shape[num],
                                            }
                if index[num] == 1:
                    print(gps_point[num])
                num += 1
                pic_dict["objects"] = objects
            json_list.append(pic_dict)
            pbar.update(1)
            # import ipdb; ipdb.set_trace();
        json.dump(json_list, a, ensure_ascii = False, indent = 2)
        pbar.close()
        # import ipdb; ipdb.set_trace();

def read_gps(save_path):
    gpss = []
    with open(save_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            data_line=line.split(' ')    
            gps =list(map(float,data_line[:]))[:]#[floatdata_line[0], data_line[1], data_line[2]]
            gpss.append(gps) 
    return gpss     

folder_path = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/tan2.json"
query_intrinsics = "/media/ubuntu/DB/target1/queries/w_intrinsic_com.txt"
render_poses = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/1_estimated_pose.txt"
render_camera = "/media/ubuntu/DB/target1/queries/w_intrinsic_com.txt"
results_position = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/pixel.json"
save_path = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/gps_pixel.txt"
save_json_path = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/output.json"
Points_3D, per_points_num, poses, name_list, shape = json_read(folder_path, query_intrinsics)

# get_pixel_gps(name_list, render_poses, render_camera, results_position)
# Pixel_Points_3D, _, _, _, _ = json_read(results_position, query_intrinsics)
# Pixel_Points_3D = np.array(Pixel_Points_3D)
# gps_pixel = cgcs2000towgs84(Pixel_Points_3D, save_path)

gps_pixel = read_gps(save_path)
in_corners = np.array(Points_3D)
# in_corners[:, 2] = 1.0
index = nms_my(in_corners.T, shape,  2) #20cm, output: [Point_NUM, ]
save_path = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/gps_imgs.txt"
gps_img = read_gps(save_path)

poses = np.array(poses)
trans_list = [pose[:3, 3] for pose in poses]

euler_img = rotation2euler(poses)#, cgcs2000towgs84(trans_list, save_path) # output: [IMG_NUM, ]

Points_3D = np.array(Points_3D)
save_path = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/gps_points.txt"
gps_point = read_gps(save_path)
# gps_point = cgcs2000towgs84(Points_3D, save_path)
update_point3D_index(save_json_path, query_intrinsics,name_list, Points_3D, index, 
                     euler_img, gps_img, gps_point, gps_pixel, per_points_num,shape)
# # json save
# # save_json(point, point_ID, per_img_num)
