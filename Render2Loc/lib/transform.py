import numpy as np
from .rotation_transformation import qvec2rotmat
import pycolmap
from pathlib import Path
from .utils.colmap import Camera

def parse_pose_list(path):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split('/')[-1]
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose() 
            
            T = np.identity(4)
            T[0:3,0:3] = R
            T[0:3,3] = -R.dot(t)  
            poses[name] = T
            
    
    assert len(poses) > 0
    return poses

def parse_db_intrinsic_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            name = data_line[0].split('/')[-1]
            _,_,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  
            
            K_w2c = np.array([ #!
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            images[name] = K_w2c
            break
  
    return K_w2c

def parse_image_list(path, with_intrinsics=False, with_colmap = False):
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            name = name.split('/')[-1]
            if with_intrinsics:
                model, width, height, *params = data
                params = np.array(params, float)
                cam = pycolmap.Camera(model, int(width), int(height), params)
                images.append((name, cam))
            elif with_colmap:
                camera_model, width, height, *params = data
                params = np.array(params, float)
                camera = Camera(
                    None, camera_model, int(width), int(height), params)
                images.append((name, camera))
            else:
                images.append(name)
    return images

def parse_image_lists(paths, with_intrinsics=False, with_colmap = False):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics, with_colmap = with_colmap)
    return images