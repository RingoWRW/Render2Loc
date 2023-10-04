import bpy 
import os 
import numpy as np
import mathutils
import sys


blend_path = str(sys.argv[-3])
input_recons = str(sys.argv[-2])
origin_coord = str(sys.argv[-1])

###############load obj    
for tile in os.listdir(input_recons):
    obj_path = os.path.join(input_recons, tile, tile+'.obj')
    bpy.ops.import_scene.obj(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.rotation_euler = (0, 0, 0)


# bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
# bpy.context.scene.display.shading.light = 'FLAT'
# bpy.context.scene.display.shading.color_type = 'TEXTURE'
# bpy.context.scene.render.image_settings.file_format = 'JPEG'
# print(blend_path) 
# bpy.ops.wm.save_mainfile(filepath = blend_path, compress = False)