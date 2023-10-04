import os
import argparse
from pathlib import Path
from  render2loc.lib import (
    localize_render2loc,
    match_feature,
    generate_seed
)
import yaml
import immatch
from hloc.visualization import visualize_from_h5
from hloc import pair_from_seed
from sensloc.utils.blender import blender_engine_phone, blender_importObjs
from hloc import  localize_sensloc, ray_casting, ray_casting_east
from sensloc.utils.preprocess import video_to_frame
from sensloc.utils.preprocess import video_to_frame, read_SRT, generate_render_pose, read_EXIF, read_gt, read_RTK_YU
from hloc import  localize_sensloc, ray_casting, localize_sensloc_loftr, localize_sensloc_loftr_withgravity,undistort

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
    "video_to_frame":{
        "input_video_path": "/media/ubuntu/DB/target_2/",
        "output_image_path": "/media/ubuntu/DB/target_2/East/images/video",
        "quries": "./datasets/Tibet/Queries/process/video/seq1"},
    "read_exif":{
        "input_EXIF_photo" :"/home/ubuntu/Documents/code/SensLoc/datasets/East/raw/images_test",
        "distortion": [
                        0.293656243361741,
                        -1.13028438807054,
                        0.000113131446409535,
                        5.29911015250079e-05,
                        1.24340747827876
                        ], 
    },
    "pairs":{

    "render_path":"wide_a",
    "query_camera":"queries/30_intrinsics.txt", 
    "reference_camera":"db_intinsics.txt", 
    "render_pose":"wide_angle/db_pose_200.txt"},

    
    
    
    "localize_render2loc":{
    "sequence":"phone_day_sequence1/", 
    "results":"./results/airloc/",
    "loc": {"estimation": {"ransac": {"max_error": 12}}, **({})}},  
    
    "blender":{
    "blender_path": "/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender",
    "rgb_path":"/home/ubuntu/Documents/1-pixel/blender_demo/east2/rgb.blend",
    "origin": "/media/ubuntu/DB/target1/metadata.xml",
    "input_recons" : "/media/ubuntu/DB/target1/Data1",
    "depth_path":"/home/ubuntu/Documents/1-pixel/blender_demo/east2/depth.blend",
    "python_rgb_path":"/home/ubuntu/Documents/code/SensLoc/sensloc/utils/blender/phone_rgb.py",
    "python_depth_path":"/home/ubuntu/Documents/code/SensLoc/sensloc/utils/blender/phone_depth.py",
    "python_importObjs_rgb_path":"/home/ubuntu/Documents/code/SensLoc/sensloc/utils/blender/importObjs_rgb.py",
    "python_importObjs_depth_path":"/home/ubuntu/Documents/code/SensLoc/sensloc/utils/blender/importObjs_depth.py",
    "f_mm": 4.5,
    "sensor_width": 6.29,
    "sensor_height": 4.71,    
    "iteration_nums":4,
    "aborlation": "fix_yaw", 
    },
    

    "ray_casting": {
        "object_name": "T-barrier0",
        "num_sample": 100,
        "DSM_path": "/media/ubuntu/DB/DSM/Production_5_DSM_merge.tif",
        "DSM_npy_path": "/media/ubuntu/DB/DSM/DSM_array.npy",
        "area_minZ": 20.580223,
        "write_path": "./predictXYZ.txt"
    },

}
# Setup the paths
dataset = args.dataset
images = dataset / 'images/video/'
query_path = dataset /'qureies'
images = dataset / 'images'
query_camera = query_path / 'intrinsics/w_intrinsic.txt'
render_camera =query_path / 'intrinsics/w_intrinsic.txt'



dsm_file = "/media/ubuntu/DB/DSM/Production_4_DSM_merge.tif"

outputs = args.outputs  # where everything will be saved
render_images = dataset / 'render_images'
render_poses = query_path / 'poses/w_pose.txt'

results_position =outputs  / 'xyz.json'


input_EXIF_photo = config['read_exif']['input_EXIF_photo']
distortion = config['read_exif']['distortion']
query_gt_xml = None



parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file',
    default='./config/config_evaluate.json',
    type=str,
    help='configuration file',
)
args = parser.parse_args()

# with open(args.config_file) as fp:
#     config = json.load(fp)

method = 'superglue'
print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visualize matches of {method}')
config_file = f'render2loc/configs/{method}.yml'

with open(config_file, 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)['airloc']
    if 'ckpt' in args:
        args['ckpt'] = os.path.join('..', args['ckpt'])
    class_name = args['class']

# Init model
model = immatch.__dict__[class_name](args)
matcher = lambda im1, im2: model.match_pairs(im1, im2)

data = dict()
# extract local features(db)
# add seed for noisy prior(outputs/seed)
# video_to_frame.main(config["video_to_frame"])
input_EXIF_photo = "/home/ubuntu/Documents/code/SensLoc/datasets/East/raw/color_images"
render_poses = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/1_estimated_pose.txt"
render_camera = "/media/ubuntu/DB/target1/queries/w_intrinsic_com.txt"
results_position = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/tan2.json"
center_points_list = read_EXIF.json_read(input_EXIF_photo)



area_minZ = ray_casting.DSM2npy(config["ray_casting"])
config["ray_casting"]["area_minZ"] = area_minZ
ray_casting_east.main_foreast_center(config["ray_casting"], render_poses, render_camera, results_position, center_points_list)
