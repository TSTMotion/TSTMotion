import numpy as np
import argparse
import pickle
import trimesh
import os
import glob
import torch
from natsort import natsorted
from trimesh import transform_points
from pyquaternion import Quaternion as Q
from scipy.spatial.transform import Rotation

import utils.configuration as config
from utils.visualization import frame2video, render_motion_in_scene
from utils.model_utils import GeometryTransformer
from utils.coordinate_system_conversion import rotate_3D_z, rotate_2D_z
import utils.rotation_conversions as geometry


def transform_scene(start, end, theta_scene, scene_x, scene_y, x_min_total, y_min_total, z_min_total):
    start[0] += x_min_total
    start[1] += y_min_total
    start[2] += z_min_total
    end[0] += x_min_total
    end[1] += y_min_total
    end[2] += z_min_total

    start = rotate_3D_z(-theta_scene,start)
    end = rotate_3D_z(-theta_scene,end)

    start[0] += scene_x
    start[1] += scene_y
    end[0] += scene_x
    end[1] += scene_y

    return start, end


def transform_motion(start,end,theta,trans,orient,motion_tendency):
    trans_start = trans[0,:].copy()
    trans -= trans_start
    trans[:,1:] = rotate_2D_z(np.pi/2, trans[:,1:])
    trans = rotate_3D_z(theta, trans)

    orient_ = np.zeros((orient.shape[0],3),dtype=np.float32)
    R1 = np.array([
        [1, 0, 0],
        [0,np.cos(np.pi/2), -np.sin(np.pi/2)],
        [0,np.sin(np.pi/2), np.cos(np.pi/2)],
    ], dtype=np.float32)
    R2 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    for i in range(len(orient)):
        orient_i = orient[i]
        # orient_i = Q(axis=orient_i/np.linalg.norm(orient_i), angle=np.linalg.norm(orient_i)).rotation_matrix
        orient_i = np.matmul(R1, orient_i)
        orient_i = np.matmul(R2, orient_i)
        # orient_i = Q(matrix=orient_i)
        # orient_i = orient_i.axis * orient_i.angle
        orient_i = geometry.matrix_to_axis_angle(torch.tensor(orient_i))
        orient_[i,:] = orient_i.numpy().astype(np.float32)

    # trans[:,0] +=  trans_start[0]
    # trans[:,1] -=  trans_start[2]
    # trans[:,2] +=  trans_start[1]

    if motion_tendency == 'toward':
        trans_end = trans[-1,:].copy()
        trans += end - trans_end
    elif motion_tendency == 'away':
        trans_start = trans[0,:].copy()
        trans += start - trans_start
    
    return trans, orient_



def visualize_result(args, p, SMPL_path, sample_index ,fps, save_folder=None, del_imgs=True):
    if args.scene_name == 'ScanNet':
        scene_id = p['scene_id']
        scene_path = os.path.join('./datasets/ScanNet_v2_raw/scans', '{}/{}_vh_clean_2.ply'.format(scene_id, scene_id))

    elif args.scene_name =='Prox':
        scene_id = p['scene_id']
        scene_path = os.path.join('./datasets/Prox_SAM3D', '{}.ply'.format(scene_id))

    elif args.scene_name =='DIMOS':
        scene_id = p['scene_id']
        scene_path = os.path.join('./datasets/DIMOS', '{}/model.ply'.format(scene_id))

    elif args.scene_name =='Demo':
        scene_path = p['scene_pc']

    # loading scence
    print(">>> Sample Index: {}".format(sample_index))
    print('>>> Loading Scence...')
    static_scene = trimesh.load(scene_path, process=False)
    theta_scene = p['theta_scene']
    scene_x =  p['scene_x']
    scene_y =  p['scene_y']
    x_min_total =  p['x_min_total']
    y_min_total =  p['y_min_total']
    z_min_total =  p['z_min_total']
    start = p['starting'][sample_index].copy()
    end = p['ending'][sample_index].copy()
    start, end = transform_scene(start, end, theta_scene, scene_x, scene_y, x_min_total, y_min_total, z_min_total)
    

    # loading motion
    print('>>> Loading Motion...')
    betas = np.zeros((10),dtype=np.float32)
    motion_tendency = p['motion_tendency']
    theta_human = p['theta_human'][sample_index] + np.pi/2 - theta_scene

    seq_len = p['root_translation'].shape[2]
    body_pose = p['rotations'][sample_index].copy().reshape(seq_len,-1)[:,:21*3]
    trans = p['root_translation'][sample_index].copy().transpose(1,0)
    orient = p['global_orient'][sample_index].copy()
    hand_pose = np.zeros((trans.shape[0],90),dtype=np.float32)
    trans, orient = transform_motion(start,end,theta_human,trans,orient,motion_tendency)
    

    ## rendering mp4
    print('>>> Rendering Video...')
    save_folder = SMPL_path.replace('motion.npy','')
    save_folder = os.path.join(save_folder,'result_{}/'.format(sample_index))
    render_motion_in_scene(
        args=args,
        smplx_folder=config.smplx_folder,
        save_folder=os.path.join(save_folder, 'imgs/'),
        scene_mesh=static_scene,
        auto_camera=False,
        num_betas=10,
        model_type='smplx',
        betas = betas,
        trans = trans,
        orient = orient,
        body_pose = body_pose,
        hand_pose = hand_pose,
    )
    frame2video(
        path=os.path.join(save_folder, 'imgs/%03d.png'),
        video=os.path.join(save_folder, 'result_{}.mp4'.format(sample_index)),
        start=0,
        framerate=fps,
    )
    if del_imgs:
        os.system('rm -rf "{}"'.format(os.path.join(save_folder, 'imgs')))
    print('\n')
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--fps', type=int)
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--output_obj', type=str)
    args = parser.parse_args()
    fps = args.fps

    for item in os.listdir(args.input_path):
        item = os.path.join(args.input_path,item)
        SMPL_path = os.path.join(item,'motion.npy')
        if not os.path.exists(SMPL_path):
            continue

        p = np.load(SMPL_path,allow_pickle=True).item() 
        sample_num = p['rotations'].shape[0]
        for sample_index in range(sample_num):
            save_folder = SMPL_path.replace('motion.npy','')
            video=os.path.join(save_folder, 'result_{}/result_{}.mp4'.format(sample_index,sample_index))
            if os.path.exists(video):
                continue
            visualize_result(args,p,SMPL_path,sample_index,fps)


if __name__ == '__main__':
    main()
