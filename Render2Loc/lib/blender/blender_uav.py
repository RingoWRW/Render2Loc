import bpy 
import os 
import numpy as np
# from pyproj import Proj
# from pyproj import Transformer
# from pyproj import CRS
from math import radians

import math
import mathutils
from mathutils import Vector
import numpy as np
import sys

def parse_render_image_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            name = (data_line[0].split('/')[-1]).split('.')[0]
            w, h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  
            
            K = [w, h,fx,fy,cx,cy]
            images[name] = K
  
    return images, int(w), int(h), int(fx), int(fy)

def parse_image_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            model, width, height, *params = data
            params = np.array(params, float)
            images[os.path.basename(name).split('.')[0]] = (model, int(width), int(height), params)
  
    assert len(images) > 0
    return images

def parse_pose_list(path, origin_coord):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = (data[0].split('/')[-1]).split('.')[0]
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose()  #c2w
            
            T = np.identity(4)
            T[0:3,0:3] = R
            T[0:3,3] = -R.dot(t)   #!  c2w
            #

            if origin_coord is not None:
                origin_coord = np.array(origin_coord)
                T[0:3,3] -= origin_coord
            transf = np.array([
                [1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1.],
            ])
            T = T @ transf
            
            poses[name] = T
            
    
    assert len(poses) > 0
    return poses

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def prepare_scene(resolution=[760, 760], device='GPU'):
    bpy.data.scenes["Scene"].render.resolution_x = resolution[0]
    bpy.data.scenes["Scene"].render.resolution_y = resolution[1]
#    bpy.data.scenes["Scene"].cycles.device = device
    try:
        delete_object('Cube')
        delete_object('Light')
    except:
        pass

def add_camera(xyz=(0, 0, 0),
               rot_vec_degree=(0, 0, 0),
               name=None,
               proj_model='PERSP',
               f=35,
               sensor_fit='HORIZONTAL',
               sensor_width=32,
               sensor_height=18,
               clip_start=0.1,
               clip_end=10000):
    bpy.ops.object.camera_add(location=xyz, rotation=rot_vec_degree)
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name
    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end
    return cam
        
def prepare_world(image_save_path, name):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    
    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    
    scene = bpy.context.scene
    cam = scene.objects['Camera']
    for output_node in [depth_file_output]: #image_file_output
        output_node.base_path = ''

    # save exr file            
    scene.render.filepath = image_save_path
    depth_file_output.file_slots[0].path = scene.render.filepath + name
    bpy.ops.render.render(write_still=True)



def delete_object(label):
    """
    Definition to delete an object from the scene.
    Parameters
    ----------
    label          : str
                     String that identifies the object to be deleted.
    """
    bpy.data.objects.remove(bpy.context.scene.objects[label], do_unlink=True)

# Clear all nodes in a mat
#!args

image_save_path = str(sys.argv[-1])
input_pose = str(sys.argv[-2])
intput_intrin = str(sys.argv[-3])
f_mm = float(sys.argv[-4])


input_recons = str(sys.argv[-6])
depth_save_path = str(sys.argv[-7])

str_origin_coord = str(sys.argv[-5])
origin_coord = [float(str_origin_coord.split(",")[0][1:]), float(str_origin_coord.split(",")[1]), float(str_origin_coord.split(",")[2][:-1])]

# GET pose
poses = parse_pose_list(input_pose, origin_coord)
intrinsics, w, h,fx, fy = parse_render_image_list(intput_intrin)

names = list(poses.keys())
# obj loader
for tile in os.listdir(input_recons):
    obj_path = os.path.join(input_recons, tile, tile+'.obj')
    bpy.ops.import_scene.obj(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.rotation_euler = (0, 0, 0)
    
# output RGB images
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'FLAT'
bpy.context.scene.display.shading.color_type = 'TEXTURE'
bpy.context.scene.render.image_settings.file_format = 'JPEG'

for name in names:
    # load pose
    pose_frame = poses[name]  
    R = pose_frame[:3, :3]
    t = list(pose_frame[:3, 3])
    rot_mat = mathutils.Matrix(list(R))
    rot_eulr = rot_mat.to_euler()   
    # initial camera
    intrin_factor_x = fx / f_mm
    intrin_factor_y = fy / f_mm
    camera_obj = add_camera(xyz = t, rot_vec_degree = rot_eulr, f=f_mm, sensor_fit = 'HORIZONTAL', sensor_width=w/intrin_factor_x, sensor_height=h/intrin_factor_y, clip_end=10000)
    # camera_obj = add_camera(xyz = t, rot_vec_degree = rot_eulr, f=f_mm,  sensor_fit='HORIZONTAL',sensor_width=sensor_width, sensor_height=sensor_height, clip_end=10000)
    bpy.context.scene.camera = camera_obj
    # initial output image
    prepare_scene(resolution=[w, h], device='GPU')
    bpy.context.scene.render.filepath = image_save_path +'/'+name.split('/')[-1]
    bpy.ops.render.render(write_still=True)  


# output depth images
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
bpy.context.scene.render.image_settings.exr_codec = 'ZIP'
bpy.context.scene.render.image_settings.use_zbuffer = True
bpy.context.scene.render.image_settings.color_depth = '32'
bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
depth_save_path = depth_save_path +'/'
for name in names:
    # load pose
    pose_frame = poses[name]  
    R = pose_frame[:3, :3]
    t = list(pose_frame[:3, 3]) 
    rot_mat = mathutils.Matrix(list(R))
    rot_eulr = rot_mat.to_euler()   
    if not os.path.exists(depth_save_path + (name.split('/')[-1]).split('.')[0] + '0001.exr'):
        # initial camera
        camera_obj = add_camera(xyz = t, rot_vec_degree = rot_eulr, f=f_mm, sensor_fit = 'HORIZONTAL', sensor_width=w/intrin_factor_x, sensor_height=h/intrin_factor_y, clip_end=10000)
        bpy.context.scene.camera = camera_obj
        # initial output image
        
        prepare_scene(resolution=[w, h], device='GPU')
        prepare_world(depth_save_path, (name.split('/')[-1]).split('.')[0])

    