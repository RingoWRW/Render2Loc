import os
import cv2
from . import logger
import time
import pycolmap
import poselib
import torch
import numpy as np
import matplotlib.cm as cm
import pickle
from .plotting import make_matching_figure
from lib.rotation_transformation import qvec2rotmat, rotmat2qvec
from tqdm import tqdm
import h5py
# from pixlib.geometry import Camera, Pose
from typing import Dict, List, Union
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def interpolate_depth(pos, depth):
    ids = torch.arange(0, pos.shape[0])
    depth = depth[:,:,0]
    h, w = depth.size()
    
    
    
    i = pos[:, 0]
    j = pos[:, 1]

    # Valid corners, check whether it is out of range
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]


    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    # vaild index
    ids = ids[valid_depth]
    
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    #depth is got from interpolation
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]

# read depth
def read_valid_depth(depth_exr,mkpts1r):
    depth = cv2.imread(str(depth_exr), cv2.IMREAD_UNCHANGED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth = torch.tensor(depth).to(device)
    mkpts1r = torch.tensor(mkpts1r).to(device)
    mkpts1r_a = torch.unsqueeze(mkpts1r[:,0],0)
    mkpts1r_b =  torch.unsqueeze(mkpts1r[:,1],0)
    mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0).to(device)

    depth, _, valid = interpolate_depth(mkpts1r_inter , depth)

    return depth.cpu(), valid

def loc_query(mkpq, mkp3d, K_w2c, w, h, max_error=5):
    height, width = int(h), int(w)
    cx = K_w2c[0][2]
    cy = K_w2c[1][2]
    focal_length = K_w2c[0][0]
    

    cfg = {
            'model': 'SIMPLE_PINHOLE',
            'width': width,
            'height': height,
            'params': [focal_length, cx, cy]
            
        }
    ret = pycolmap.absolute_pose_estimation(
            mkpq.numpy(), mkp3d.numpy(), cfg, max_error)

    return ret

def loc_query_wa(mkpq, mkp3d, K_w2c, w, h, max_error=5):
    height, width = int(h), int(w)
    cx = K_w2c[0][2]
    cy = K_w2c[1][2]
    fx = K_w2c[0][0]
    fy = K_w2c[1][1]
    for i in range(0, mkpq.shape[0]):
        keypoint = mkpq
    
    
    
    cfg = {
            'model': 'PINHOLE',
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy]
            
        }
    
    ret = pycolmap.absolute_pose_estimation(
            mkpq.numpy(), mkp3d.numpy(), cfg, max_error)

    return ret

def Get_Points3D(depth, R, t, K, points):   # points[n,2]
    '''
    depth, R, t, K, points: toprch.tensor
    return Point3D [n,3]: numpy.array
    '''
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[ :, [0]])], dim=-1)
        points_2D = points_2D.T  
    t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
    Points_3D = R @ K @ (depth * points_2D) + t   
    return (Points_3D.T).numpy().astype(np.float64)    #[3,n]

def blender_engine(blender_path, project_path, script_path, intrinscs_path, extrinsics_path, image_save_path):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path, 
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    cmd = '{} -b {} -P {} -- {} {} {}'.format(blender_path, 
                                            project_path, 
                                            script_path, 
                                            intrinscs_path,
                                            extrinsics_path, 
                                            image_save_path,
                                )
    os.system(cmd)           

class QueryLocalizer:
    def __init__(self, config=None):
        self.config = config or {}

    def localize(self, points3D, points2D, query_camera):
        points3D = [points3D[i] for i in range(points3D.shape[0])]
        # import ipdb; ipdb.set_trace();
        fx, fy, cx, cy = query_camera.params
        cfg = {
            "model": "PINHOLE",
            "width": query_camera.width,
            "height": query_camera.height,
            "params": [fx, fy, cx, cy],
        }  
        
        # pose, info = poselib.estimate_absolute_pose(
        #     points2D, points3D, cfg,
        #     {
        #         'max_error': float(5.0),
        #         'use_gravity_axis': False,
        #         'gravity_axis': np.array([0, 0, 0]),
        #         'gravity_thresh': 1
        #     },
        #     {}
        # )

        # q, t = (rotmat2qvec(pose[:, :3]), pose[:, 3])
        # R = np.asmatrix(pose[:, :3]).transpose()
        # t = -R.dot(t)
        

        
        ret = pycolmap.absolute_pose_estimation(
            points2D,
            points3D,
            cfg,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )
        # import ipdb; ipdb.set_trace();
        
        return ret  

def main(config, 
         data, 
         iter,
         outputs,
         con: Dict = None,
         ):
    save_loc_path = outputs / (str(iter) + "_estimated_pose.txt")
    sequence =config["sequence"]
    
    iterative_num = data["iter"]
    queries= data["quries"]
    K_r= data["render_intrinsics"]
    matches = data["matches"]
    
    query_names = data["query_name"]
    render_poses = data["render_pose"]
    render_names = data["render_name"]
    

    
    pbar = tqdm(total=len(query_names), unit='pts')
    con = {"estimation": {"ransac": {"max_error": int(12)}}, **(con or {})}
    localizer = QueryLocalizer(con)   
    poses = {}

    logger.info('Starting localization...')
    t_start = time.time()
    for qname, query_camera in tqdm(queries):  
        print(qname)
        match  = matches[qname.split('/')[-1]]
        imgr_name = match["imgr_name"]
        depth_exr_final = match["exrr_pth"]
        max_correct = match["correct"]
        mkpts_r = match["mkpts_r"]
        mkpts_q = match["mkpts_q"]
        if depth_exr_final is None or max_correct == 0:
            print("no match")
            qvec, tvec = 0, 0
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = 'render'+str(iterative_num)+'/'+sequence+qname.split('.')[0] + '.png'
            f.write(f'{name} {qvec} {tvec}\n') 
        else:     
            # get 3D Points 
            K_w2c = torch.tensor(K_r).float()
            K_c2w = K_w2c.inverse()
            depth, valid=read_valid_depth(depth_exr_final,mkpts_r)  
            render_pose = torch.tensor(render_poses[imgr_name]).float()
            Points_3D = Get_Points3D(
                depth,
                render_pose[:3, :3],
                render_pose[:3, 3],
                K_c2w,
                torch.tensor(mkpts_r),
            )


            # pnp
            points2D = mkpts_q[valid].cpu().numpy()
            num_matches = points2D.shape[0]
            ret = localizer.localize(Points_3D, points2D, query_camera)
            ret['camera'] = {
                'model': query_camera.model_name,
                'width': query_camera.width,
                'height': query_camera.height,
                'params': query_camera.params,
            }
            log = {
                'PnP_ret': ret,
                'keypoints_query': points2D,
                'points3D_ids': None,
                'points3D_xyz': None,  # we don't log xyz anymore because of file size
                'num_matches': num_matches,
                'keypoint_index_to_db': None,
            }
            pbar.update(1)
        if ret['success']:
            poses[qname] = (ret['qvec'], ret['tvec'])
        t_end = time.time()
        logger.info(f'Localize uses {t_end-t_start} seconds.')
        logger.info(f'Localized {len(poses)} / {len(queries)} images.')
        logger.info(f'Writing poses to {save_loc_path}...')

        with open(save_loc_path, 'w') as f:
            for q in poses:
                qvec, tvec = poses[q]
                qvec = ' '.join(map(str, qvec))
                tvec = ' '.join(map(str, tvec))
                name = q.split('/')[-1]
                f.write(f'{name} {qvec} {tvec}\n')

        logs_path = f'{save_loc_path}_logs.pkl'
        logger.info(f'Writing logs to {logs_path}...')
        logger.info('Done!')               
           
        pbar.close()  

    # update pose 
    return save_loc_path


if __name__=="__main__":
    main()


