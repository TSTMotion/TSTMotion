import os
import argparse
import pickle
import sys
import re
import json
import random

import numpy as np
import torch
import openai
import trimesh

sys.path.append('./OmniControl')
from OmniControl.utils.parser_util import add_base_options,add_sampling_options,add_edit_options,add_generate_options
from OmniControl.utils.parser_util import add_data_options,add_model_options,add_diffusion_options
from OmniControl.sample.generate import generate_motion
from OmniControl.utils import dist_util
from OmniControl.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from OmniControl.data_loaders.get_data import get_dataset_loader
from OmniControl.visualize.vis_utils import npy2obj
from utils.coordinate_system_conversion import rotate_scene, scene_to_GPT, GPT_to_MDM, GPT_to_scene, MDM_to_GPT
from utils.chat_with_LLMs import get_gpt_plan, get_keyframes, get_keyjoints, get_check_plan

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prompt_path', type=str)
    parser.add_argument('--GPT_version', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--scene_pc', type=str)
    parser.add_argument('--scene_seg', type=str)
    parser.add_argument('--caption', type=str)
    parser.add_argument('--scale_lambda', type=float,default=2)
    parser.add_argument('--scale_eta', type=float,default=0.5) 

    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_generate_options(parser)

    args = parser.parse_args()
    openai.api_key = args.api_key
    print("Arguments:")
    print(args,'\n')

    return args


def foundation_generation():
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    dist_util.setup_dist(0)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='train')  # in train mode, you get both text and motion.

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model.to(dist_util.dev())
    model.eval()  # disable random masking

    return data,model,diffusion


def get_coord(scene_pc):
    scene = trimesh.load(scene_pc, process=False)
    scene = np.array(scene.vertices)
    scene, scene_x, scene_y, theta_scene, x_min_total, y_min_total, z_min_total = rotate_scene(scene)
    return scene, scene_x, scene_y, theta_scene, x_min_total, y_min_total, z_min_total


def get_segment_label(scene_seg):
    with open(scene_seg, 'rb') as file:
        seg_result = pickle.load(file)
    segment = seg_result[0]
    label = seg_result[1]

    if '' in label:
        index = []
        for i in range(len(label)):
            if label[i] != '':
                index.append(i)
        segment = segment[:,index]

        while '' in label:
            label.remove('')

    return segment,label


# The resulting bounding box will be affected by the segmentation effect and may not accurately contain the corresponding object.
def get_bounding_boxes_from_seg(scene_pc, scene_seg):
    coord, scene_x, scene_y, theta_scene, x_min_total, y_min_total, z_min_total = get_coord(scene_pc)
    segment,label = get_segment_label(scene_seg)

    bounding_boxes = []
    for i in range(len(label)):
        box_label = label[i]
        if box_label == 'wall' or box_label == 'floor' or box_label == 'rug' or box_label == 'ceiling' or box_label == 'person' or box_label == 'human' or box_label == 'lamp' :
            continue

        box_seg = coord[segment[:,i]]
        nums_pointcloud = box_seg.shape[0]
        truncated_index = int(0.01*nums_pointcloud)
        x_max,x_min = np.sort(box_seg[:,0])[[-truncated_index,truncated_index]]
        y_max,y_min = np.sort(box_seg[:,1])[[-truncated_index,truncated_index]]
        z_max,z_min = np.sort(box_seg[:,2])[[-truncated_index,truncated_index]]
        if z_min > 1.5:
            continue

        box = {
            'box_number':i,
            'label':box_label,
            'midpoint': [(x_min+x_max)/2,(y_min+y_max)/2,(z_min+z_max)/2],
            'x_min':x_min,
            'x_max':x_max,
            'y_min':y_min,
            'y_max':y_max,
            'z_min':z_min,
            'z_max':z_max,
        }
        bounding_boxes.append(box)
    
    return coord, segment, bounding_boxes, scene_x, scene_y, theta_scene, x_min_total, y_min_total, z_min_total


def post_check(args,all_motions,data,model,diffusion,caption,keyframes,result_path,coord,target,plan):
    check_plan = get_check_plan(args,all_motions,caption,coord,target,plan,keyframes,result_path)
    if check_plan==plan:
        theta_human, plan_in_MDM, starting, ending = GPT_to_MDM(plan)
        return all_motions, starting, ending, theta_human, False
    else:
        theta_human, plan_in_MDM, starting, ending = GPT_to_MDM(check_plan)
        keyframes = get_keyframes(plan_in_MDM)
        keyjoints,keyjoints_loc = get_keyjoints(plan_in_MDM)
        all_motions = generate_motion(args,data,model,diffusion,caption,keyframes,keyjoints,keyjoints_loc)
        return all_motions, starting, ending, theta_human, True


def post_check_loop(args,all_motions,data,model,diffusion,caption,keyframes,result_path,coord,target,plan,starting,ending,theta_human):
    all_motions = MDM_to_GPT(plan['motion_tendency'],all_motions,starting,ending,theta_human) # (batch, joint(22), 3, frame)

    loop_count = 0
    loop_max = 1
    loop_tag = True
    while loop_count<loop_max and loop_tag==True:
        loop_count += 1
        all_motions, starting, ending, theta_human, loop_tag = post_check(args,all_motions,data,model,diffusion,caption,keyframes,result_path,coord,target,plan)
        
    return all_motions, starting, ending, theta_human


def save_result(args, result_path, scene_pc, caption, 
                    plan, all_motions, theta_human, starting, ending, 
                    scene_x, scene_y, theta_scene, x_min_total, y_min_total, z_min_total):
    starting, ending = GPT_to_scene(starting, ending)
    my_npy2obj = npy2obj(motion=all_motions, device=args.device)
    root_translation,global_orient,rotations = my_npy2obj.get_smpl_params_by_batch()
    # my_npy2obj.save_obj(result_path)

    npy_path = os.path.join(result_path, 'motion.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {"all_motions":all_motions,'caption': caption, "scene_pc":scene_pc,'text_condition':args.text_condition,
             'scene_x':scene_x,'scene_y':scene_y, 'theta_scene':theta_scene, 'x_min_total':x_min_total, 'y_min_total':y_min_total, 'z_min_total':z_min_total,
             'theta_human':[theta_human]*args.num_samples, 'starting':[starting]*args.num_samples, 'ending':[ending]*args.num_samples, 
             'motion_tendency':plan['motion_tendency'], 'root_translation':root_translation, 'global_orient':global_orient, 'rotations':rotations, 
             })
    

if __name__ == '__main__':
    args = get_args()
    data,model,diffusion = foundation_generation()
    scene_name, scene_pc, scene_seg, caption = args.scene_name, args.scene_pc, args.scene_seg, args.caption
    result_path = os.path.join(scene_name, caption.replace(' ','_'))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ## Input to TSTMotion
    coord, segment, bounding_boxes, scene_x, scene_y, theta_scene, x_min_total, y_min_total, z_min_total = get_bounding_boxes_from_seg(scene_pc, scene_seg)
    bounding_boxes = scene_to_GPT(bounding_boxes)

    ## Output from TSTMotion
    target, caption, plan = get_gpt_plan(args,coord,segment,caption,bounding_boxes,result_path)
    theta_human, plan_in_MDM, starting, ending = GPT_to_MDM(plan)
    keyframes = get_keyframes(plan_in_MDM)
    keyjoints,keyjoints_loc = get_keyjoints(plan_in_MDM)
    all_motions = generate_motion(args,data,model,diffusion,caption,keyframes,keyjoints,keyjoints_loc) # (batch, joint(22), 3, frame)
    all_motions, starting, ending, theta_human = post_check_loop(args,all_motions,data,model,diffusion,caption,keyframes,result_path,coord,target,plan,starting,ending,theta_human)
    save_result(args, result_path, scene_pc, caption, 
                plan, all_motions, theta_human, starting, ending, 
                scene_x, scene_y, theta_scene, x_min_total, y_min_total, z_min_total)







