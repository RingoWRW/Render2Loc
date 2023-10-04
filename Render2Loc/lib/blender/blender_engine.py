import os
from pathlib import Path

def blender_engine_old(
    blender_path,
    project_path,
    script_path,
    origin,  
    intrinscs_path,
    extrinsics_path,
    image_save_path,
):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path,
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    # cmd = '{} -b {} -P {} -- {} {} {}'.format(
    #     blender_path,
    #     project_path,
    #     script_path,
    #     intrinscs_path,
    #     extrinsics_path,
    #     image_save_path,  
    # )
    cmd = '{} -b {} -P {} -- {} {}  {} {}'.format(
        blender_path,
        project_path,
        script_path,
        origin,
        intrinscs_path,
        extrinsics_path,
        image_save_path,  
    )
    os.system(cmd)

def blender_engine(
    blender_path,
    project_path,
    script_path,
    origin,
    f_mm,
    intrinscs_path,
    extrinsics_path,
    image_save_path,
    input_Objs,
    depth_save_path
):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path,
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    cmd = '{} -b {} -P {} -- {} {} {}  {} {} {} {}'.format(
        blender_path,
        project_path,
        script_path,
        depth_save_path, 
        input_Objs ,
        origin,
        f_mm,
        intrinscs_path,
        extrinsics_path,
        image_save_path, 
        
    )
    os.system(cmd)


def import_Objs(
    blender_path,
    project_path,
    script_path,
    input_Objs,
    origin,

):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path,
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    cmd = '{} -b {} -P {} -- {} {}'.format(
        blender_path,
        project_path,
        script_path,
        input_Objs,
        origin
    )
    os.system(cmd)

def main(config, 
         intrinscs_path: Path,
         extinsics_path: Path,
         type:str,
         ):
    
    rgb_save_path = config["rgb_image"]
    depth_save_path = config["depth_image"]
    blend_path = config["rgb_path"]
    blender_path = config["blender_path"]
    origin = config["origin"]
    input_Objs = config["input_recons"]
    python_obj_path = config["python_obj_path"]

    if type == "phone":
        f_mm = config["f_mm_phone"]
        python_path = config["python_path_phone"]
    elif type == "uav":
        f_mm = config["f_mm_uav"]
        python_path = config["python_path_uav"]

    print("render....")

    # render rgb and depth
    blender_engine(
        blender_path,
        blend_path,
        python_path,
        origin,
        f_mm,
        str(intrinscs_path),
        str(extinsics_path),
        rgb_save_path,
        input_Objs,
        depth_save_path
    )

   

if __name__ == "__main__":
    blender_path = ""
    sensor_height = 4.71
    sensor_width = 6.29
    f_mm = 4.5
    origin = ""