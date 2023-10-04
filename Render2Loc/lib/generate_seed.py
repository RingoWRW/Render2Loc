from osgeo import gdal
import numpy as np
import math
from .rotation_transformation import matrix_to_euler_angles, qvec2rotmat,matrix_to_quaternion,euler_angles_to_matrix,quaternion_to_matrix,rotmat2qvec
import torch
from scipy.spatial.transform import Rotation as R
import os
import glob
import numpy as np
from typing import Dict, List, Union, Optional
from pathlib import Path
def parse_pose_euler(dsm_filepath, save_filepath, iphone_prior_output):
    with open(iphone_prior_output, 'r') as f:
        for data in f.read().rstrip().split('\n'):
                data = data.split()
                name = data[0]
                q, t = np.split(np.array(data[1:], float), [4])
                q = torch.from_numpy(q)
                # R = np.asmatrix(qvec2rotmat(q))#.transpose()  #c2w
                # rotation = torch.from_numpy(R)
                rotation = quaternion_to_matrix(q)
                euler = matrix_to_euler_angles(rotation, "XYZ")

                matrix = euler_angles_to_matrix(euler, "XYZ")
                quat = matrix_to_quaternion(matrix)
                print(quat)


def query_add_seed(dsm_filepath, save_filepath, query_sequence, dev = 'phone'):
    dataset = gdal.Open(dsm_filepath)  # dsm
    
    geotrans = dataset.GetGeoTransform()
    originX = geotrans[0]
    originY = geotrans[3]
    pixelWidth = geotrans[1]
    pixelHeight = geotrans[5]
    band = dataset.GetRasterBand(1)

    delta = [[0, 0], [0, 5], [0, -5], [5, 0], [-5, 0]]
    
    with open(query_sequence, 'r') as f:
        with open(save_filepath,'w') as file_w:
            for data in f.read().rstrip().split('\n'):
                index = 0
                data = data.split()
                name_raw = data[0].split('/')[-1]
                q, t = np.split(np.array(data[1:], float), [4])       

                ### add seed
                qmat = qvec2rotmat(q)
                t = -qmat.T @ t
                qmat = qmat.T
                ### d = 1.5m
                x = t[0]
                y = t[1]
                qvec = rotmat2qvec(qmat)  #!c2w
                        #==== calculate z
                for i in range(len(delta)):
                    x = t[0] + delta[i][0]
                    y = t[1] + delta[i][1]
                    if dev == 'phone':
                        xOffset = int((x - originX) / pixelWidth)
                        yOffset = int((y - originY) / pixelHeight)
                        z = band.ReadAsArray(xOffset, yOffset, 1, 1) + 1.5  #z = 1.5
                        z = z[0][0]
                    
                        
                        # ==== q, q+60, q -60
                        qvec1, qvec2 = yaw_seed(qvec)  #! output wxyz
                        t_c2w = np.array([x, y, z])
                        x, y, z = transform_t(q ,t_c2w)
                        name = name_raw[:-4] + '_' +str(index) +'.jpg'
                        out_line_str  = name_raw[:-4]+'/' + name+' '+str(q[0])+' '+str(q[1])+' '+str(q[2])+' '+str(q[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)
                        index += 1
                        x, y, z = transform_t(qvec1 ,t_c2w)
                        name = name_raw[:-4] + '_' +str(index) +'.jpg'
                        out_line_str  = name_raw[:-4]+'/'+name + ' '+str(qvec1[0])+' '+str(qvec1[1])+' '+str(qvec1[2])+' '+str(qvec1[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)
                        index += 1
                        x, y, z = transform_t(qvec2 ,t_c2w)
                        name = name_raw[:-4] + '_' +str(index) +'.jpg'
                        out_line_str  = name_raw[:-4]+'/'+name + ' '+str(qvec2[0])+' '+str(qvec2[1])+' '+str(qvec2[2])+' '+str(qvec2[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)
                        index += 1
                    else:
                        z = t[2]
                        t_c2w = np.array([x, y, z])
                        x, y, z = transform_t(q,t_c2w)
                        name = name_raw[:-4] +'.jpg'
                        out_line_str  = name_raw[:-4]+'/' + name+' '+str(q[0])+' '+str(q[1])+' '+str(q[2])+' '+str(q[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)                        
                        

                    
    print("Done with writting pose.txt")       
def transform_t(q, t_c2w):
    
    R = np.asmatrix(qvec2rotmat(q))   
    t = -R @ t_c2w 
    t = np.array(t)
    x, y, z = t[0][0], t[0][1], t[0][2]
    return x, y, z       
def yaw_seed(qvec):
    
    qv = [ float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])]
    
    ret = R.from_quat(qv)
    euler_xyz = ret.as_euler('xyz', degrees=True)
    
    euler_xyz_2 = ret.as_euler('xyz', degrees=True)

    euler_xyz[2] = euler_xyz[2] + 30
    euler_xyz_2[2] = euler_xyz_2[2] - 30
    
    # # euler to matrix
    ret_1 = R.from_euler('xyz', euler_xyz, degrees=True)
    ret_2 = R.from_euler('xyz', euler_xyz_2, degrees=True)
    new_matrix1 = ret_1.as_matrix()
    new_matrix2 = ret_2.as_matrix()
    
    new_qvec1 = rotmat2qvec(new_matrix1.T)
    new_qvec2 = rotmat2qvec(new_matrix2.T)  
    return new_qvec1, new_qvec2

def main(
         prior_path: Path,
         seed_path: Path,
         dsm_file,
         dev = ''
         ):
    if not os.path.exists(seed_path):
        query_add_seed(dsm_file, seed_path, prior_path, dev)

