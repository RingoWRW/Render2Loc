import os
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from  lib import (
    localize_render2loc,
    match_feature,
    generate_seed,
    pair_from_seed
)
import yaml
import immatch
import json
from lib.blender import blender_engine

parser = argparse.ArgumentParser()
parser.add_argument('--outputs', type=Path, default='./outputs',#!
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--config_file',default='/media/guan/3CD61590D6154C10/SomeCodes/Render2Loc/configs/pipeline_image.json', type=str, help='configuration file')
args = parser.parse_args()

outputs = args.outputs

if not os.path.exists(outputs):
    os.makedirs(outputs) 

method = 'superglue'

with open(args.config_file) as fp:
        config = json.load(fp)


images = config["query"]["images"]
query_camera =  config["query"]["intrinsic"]
render_camera = config["query"]["intrinsic"]
prior_pose = config["query"]["prior_pose"]
render_pose = outputs / 'render_pose.txt'

dsm_file = config["model"]["DSM_file"]
render_images = config["blender"]["render_image"]


print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visualize matches of {method}')
config_file = f'/media/guan/3CD61590D6154C10/SomeCodes/Render2Loc/configs/{method}.yml'

with open(config_file, 'r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)['airloc']
    if 'ckpt' in args:
        args['ckpt'] = os.path.join('..', args['ckpt'])
    class_name = args['class']

# Init model
model = immatch.__dict__[class_name](args)
matcher = lambda im1, im2: model.match_pairs(im1, im2)

data = dict()

# load .obj  in .blend
# render rgb and depth images
generate_seed.main(prior_pose, render_pose, dsm_file, 'phone')
blender_engine.main(config["blender"], 
                    render_camera, 
                    render_pose, 
                    'phone')
# seed_pose
data = pair_from_seed.main(
                           images, 
                           render_images, 
                           query_camera,
                           render_camera,
                           render_pose,
                           data, 0)
# generate  2D match
loc_matches = match_feature.main(
                            data, 
                            matcher,
                            outputs
)
# localize, compute pose
render_poses = localize_render2loc.main(config["render2loc"], data, 1, outputs)  #!update pose file


# set iteration, more accuracy localization results
for iter in range(1, 4):
    data = pair_from_seed.main(
                           images, 
                           render_images, 
                           query_camera,
                           render_camera,
                           render_poses,
                           data, iter)
    blender_engine.main(config["blender"], 
                        render_camera, 
                        render_poses, 
                         'phone')
    loc_matches = match_feature.main(
                                data, 
                                matcher,
                                outputs
    )
    render_poses = localize_render2loc.main(config["localize_render2loc"], data, iter, outputs)  #!update pose file

# eval results




