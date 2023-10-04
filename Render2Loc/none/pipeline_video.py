import os
import argparse
from pathlib import Path
from  render2loc.lib import (
    localize_render2loc,
    match_feature,
    generate_seed
)
import json
import yaml
import immatch
from hloc.visualization import visualize_from_h5
from hloc import pair_from_seed
from sensloc.utils.blender import blender_engine_phone, blender_importObjs
from hloc import  localize_sensloc, ray_casting, ray_casting_east
from sensloc.utils.preprocess import video_to_frame

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default='datasets/East',#!
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default='outputs/render2loc/east_video',#!
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=20,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=1,
                    help='Number of image pairs for loc, default: %(default)s')
parser.add_argument('--config_file',default='render2loc/config/pipeline_rc_east_video.json', type=str, help='configuration file')
args = parser.parse_args()

with open(args.config_file) as fp:
        config = json.load(fp)

# Setup the paths
dataset = args.dataset
images = dataset / 'images/video/'
query_camera = dataset / 'queries_intrinsics/intrinsic.txt'
render_camera = dataset /'queries_intrinsics/ref_intrinsic.txt'


sensor_prior = dataset/ 'sensors_prior/pose.txt'
dsm_file = "/media/ubuntu/DB/DSM/Production_4_DSM_merge.tif"

outputs = args.outputs  # where everything will be saved
render_images = outputs / 'pic'
render_poses = outputs / 'pose.txt'


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


# video_to_frame.main(config["video_to_frame"])
if not os.path.exists(outputs):
    os.makedirs(outputs) 


# TODO: judge whether has import obj and generate seed,  pairs.txt:query path  /phone/sequences; 0001.exr
blender_importObjs.main(config["blender"])
generate_seed.main(sensor_prior, render_poses, dsm_file, 'UAV')
blender_engine_phone.main(config["blender"], 
                    render_camera, 
                    render_poses, 
                    render_images)
data = pair_from_seed.main(
                           images, 
                           render_images, 
                           query_camera,
                           render_camera,
                           render_poses,
                           data, 0)

loc_matches = match_feature.main(
                            data, 
                            matcher,
                            outputs
)
render_poses = localize_render2loc.main(config["localize_render2loc"], data, 0, outputs)  #!update pose file

for iter in range(1, 4):
    data = pair_from_seed.main(images, 
                            render_images, 
                            query_camera,
                            render_camera,
                            render_poses,
                            data, 
                            iter)

    blender_engine_phone.main(config["blender"], 
                        render_camera, 
                        render_poses, 
                        render_images)

    data = match_feature.main(data, 
                            matcher,
                            outputs)

    render_poses = localize_render2loc.main(config["localize_render2loc"], data, iter, outputs)  #!update pose file

# ---- optical target localization -----------
# center_points_list = {'DJI_20230517172102_0258_W': [{'T-barrier0': [1494.8387096774195, 787.0]}, {'T-barrier1': [1286.774193548387, 862.8064516129032]}, {'T-barrier2': [1635.1612903225805, 982.1612903225807]}, {'T-barrier3': [1452.9032258064517, 1025.7096774193549]}, {'T-barrier4': [1225.483870967742, 1111.1935483870968]}, {'T-barrier5': [1581.9354838709678, 1212.8064516129032]}, {'T-barrier6': [1165.8064516129034, 1367.6451612903224]}, {'T-barrier7': [1528.7096774193549, 1454.741935483871]}, {'T-barrier8': [1338.3870967741937, 1527.3225806451615]}, {'T-barrier9': [1470.6451612903224, 1685.3870967741937]}, {'T-barrier10': [1119.032258064516, 1609.5806451612902]}, {'T-barrier11': [1056.1290322580644, 1870.8709677419356]}, {'T-barrier12': [1272.2580645161293, 1774.0967741935483]}, {'T-barrier13': [1404.516129032258, 1961.1935483870968]}, {'T-barrier14': [1212.5806451612902, 2046.6774193548388]}, {'T-barrier15': [1354.516129032258, 2185.387096774193]}, {'T-barrier16': [998.0645161290323, 2104.741935483871]}, {'T-barrier17': [1164.1935483870968, 2278.935483870968]}, {'T-barrier18': [935.1612903225807, 2325.709677419355]}, {'T-barrier19': [1291.6129032258063, 2437.0]}, {'T-barrier20': [1117.4193548387098, 2511.1935483870966]}, {'Triangle-barrier21': [3980.3225806451615, 2077.322580645161]}, {'Triangle-barrier22': [4014.1935483870966, 2132.1612903225805]}, {'T-barrier23': [1377.0967741935483, 1298.2903225806451]}, {'T-barrier（Down）24': [862.5806451612904, 2525.709677419355]}, {'T-barrier（Down）25': [1223.8709677419356, 2572.483870967742]}, {'T-barrier（Down）26': [790.0, 2791.8387096774195]}, {'T-barrier（Down）27': [825.483870967742, 2904.741935483871]}], 'DJI_20230518121639_0393_W': [{'Rectangle-barrier0': [1862.5806451612902, 1388.6129032258063]}, {'Rectangle-barrier1': [1856.1290322580646, 1503.1290322580644]}, {'Rectangle-barrier2': [1860.967741935484, 1627.3225806451612]}, {'Rectangle-barrier3': [1738.3870967741937, 1683.7741935483873]}, {'Rectangle-barrier4': [1617.4193548387098, 1717.6451612903227]}, {'Rectangle-barrier5': [1857.741935483871, 1737.0]}, {'Rectangle-barrier6': [1627.0967741935483, 1837.0]}, {'Rectangle-barrier7': [1854.516129032258, 1840.225806451613]}, {'Rectangle-barrier8': [1727.0967741935483, 1912.8064516129034]}, {'Rectangle-barrier9': [1622.258064516129, 1987.0]}, {'Rectangle-barrier10': [1849.6774193548388, 1975.7096774193549]}, {'Rectangle-barrier11': [1722.258064516129, 2041.8387096774195]}, {'Rectangle-barrier12': [1851.2903225806451, 2098.290322580645]}, {'Rectangle-barrier13': [1619.032258064516, 2106.354838709677]}, {'Rectangle-barrier14': [1722.2580645161293, 2164.4193548387098]}, {'Rectangle-barrier15': [1614.1935483870966, 2257.967741935484]}, {'Rectangle-barrier16': [1856.1290322580644, 2233.7741935483873]}, {'Rectangle-barrier17': [1719.032258064516, 2307.967741935484]}, {'Rectangle-barrier18': [1849.6774193548388, 2348.290322580645]}, {'Rectangle-barrier19': [1614.1935483870966, 2354.741935483871]}, {'Rectangle-barrier20': [1717.4193548387098, 2414.4193548387093]}, {'Rectangle-barrier21': [1848.0645161290324, 2472.483870967742]}, {'Rectangle-barrier22': [1610.967741935484, 2482.1612903225805]}, {'Rectangle-barrier23': [1719.032258064516, 2537.0]}, {'Rectangle-barrier24': [1849.6774193548388, 2598.290322580645]}, {'Rectangle-barrier25': [1609.3548387096776, 2593.451612903226]}, {'Rectangle-barrier26': [1728.7096774193549, 2657.967741935484]}, {'Rectangle-barrier27': [1852.9032258064517, 2707.967741935484]}, {'Rectangle-barrier28': [1604.516129032258, 2712.8064516129034]}, {'Rectangle-barrier29': [448.06451612903226, 1149.9032258064517]}, {'Rectangle-barrier30': [427.0967741935484, 1570.8709677419356]}, {'Rectangle-barrier31': [443.2258064516129, 1712.8064516129034]}, {'Rectangle-barrier32': [478.7096774193549, 2007.967741935484]}, {'Rectangle-barrier33': [415.80645161290323, 1983.7741935483873]}, {'Rectangle-barrier34': [430.3225806451613, 2103.1290322580644]}, {'Rectangle-barrier35': [433.5483870967742, 2157.967741935484]}, {'Rectangle-barrier36': [433.54838709677415, 2212.8064516129034]}, {'Rectangle-barrier37': [431.93548387096774, 2266.032258064516]}, {'Rectangle-barrier38': [433.5483870967742, 2325.709677419355]}, {'Rectangle-barrier39': [433.5483870967742, 2377.322580645161]}, {'Triangle-barrier40': [1120.6451612903224, 2437.0]}, {'Triangle-barrier41': [1269.032258064516, 2554.741935483871]}, {'Rectangle-barrier42': [1375.483870967742, 2190.2258064516127]}]}

# results_position = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east/XYZ.txt"
# results = "/home/ubuntu/Documents/code/SensLoc/outputs/render2loc/east/0_estimated_pose.txt"
# area_minZ = ray_casting.DSM2npy(config["ray_casting"])
# config["ray_casting"]["area_minZ"] = area_minZ
# ray_casting_east.main(config["ray_casting"], results, render_camera, results_position, center_points_list)
#     # ============= eval
    # eval.main(config["evaluate"])
