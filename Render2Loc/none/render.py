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
parser.add_argument('--dataset', type=Path, default='/media/ubuntu/DB/target1',#!
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='outputs/render2loc/east_412',#!
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
    "rgb_path":"/home/ubuntu/Documents/1-pixel/blender_demo/east/rgb.blend",
    "origin": "/media/ubuntu/DB/target1/metadata.xml",
    "input_recons" : "/media/ubuntu/DB/target1/Data1",
    "depth_path":"/home/ubuntu/Documents/1-pixel/blender_demo/east/depth.blend",
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
query_camera = dataset / 'queries/w_intrinsic_com.txt'
render_camera = dataset / 'queries/w_intrinsic_com.txt'



dsm_file = "/media/ubuntu/DB/DSM/Production_4_DSM_merge.tif"

outputs = args.outputs  # where everything will be saved
render_images = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/render_images"
render_poses = '/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/1_estimated_pose.txt'

output_images = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east_412/render_images"


results = outputs / f'WideAngle_hloc_spp_spg_netvlad_{args.num_loc}.txt'  #!




'''
seed(outputs/query_poses)
render images(render_upright)\
pairs(outputs/pairs)
features(outputs/features.h5)
matches(matches()) no


'''



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
distortion =  [
    0.293656243361741,
    -1.13028438807054,
    0.000113131446409535,
    5.29911015250079e-05,
    1.24340747827876
]
data = dict()
# extract local features(db)
# add seed for noisy prior(outputs/seed)
# video_to_frame.main(config["video_to_frame"])
input_EXIF_photo = "/home/ubuntu/Documents/code/SensLoc/datasets/East/raw/color_images"
query_gt_xml = None
query_path = "/home/ubuntu/Documents/code/SensLoc/datasets/East/raw/queries"
images = Path("/home/ubuntu/Documents/code/SensLoc/datasets/East/raw/queries/images")
query_intrinsics = Path(query_path) / 'intrinsics/w_intrinsic.txt'
# read_EXIF.for_east(input_EXIF_photo, query_gt_xml, query_path)
# undistort.main(input_EXIF_photo, images_query_w, None, query_intrinsics, distortion)
if not os.path.exists(outputs):
    os.makedirs(outputs) 
# TODO: judge whether has import obj and generate seed,  pairs.txt:query path  /phone/sequences; 0001.exr
# blender_importObjs.main(config["blender"])
blender_engine_phone.main(config["blender"], 
                    render_camera, 
                    render_poses, 
                    render_images)

