from OmniControl.visualize.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from OmniControl.visualize.simplify_loc2rot import joints2smpl


class npy2obj:
    def __init__(self, motion, device=0, cuda=True,):
        self.device = device
        self.cuda = cuda
        self.motions = motion
        self.rot2xyz = Rotation2xyz(device='cuda:'+str(device),get_obj=1)
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions.shape
        self.num_frames = self.motions[0].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=self.device, cuda=self.cuda)

        self.root_translation = []
        self.rotations = []
        self.global_orient = []
        self.vertices = []


    def get_smpl_params_by_batch(self):
        print('\n')
        for i in range(self.bs):
            print('>> get smpl params:{}'.format(i))
            self.opt_cache = {}

            motion_tensor = self.j2s.joint2smpl(self.motions[i].transpose(2, 0, 1))  # [nframes, njoints, 3]
            motion_tensor = motion_tensor.to('cuda:'+str(self.device))
            vertices,rotations,global_orient = self.rot2xyz(motion_tensor, mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        # jointstype='smpl',  # for joint locations
                                        vertstrans=True,get_rotations_back=True)

            self.root_translation.append(motion_tensor[0, -1, :3, :self.num_frames])
            self.rotations.append(rotations)
            self.global_orient.append(global_orient)
            self.vertices.append(vertices)

        self.root_translation = torch.stack(self.root_translation,dim=0).cpu().numpy()
        self.global_orient = torch.stack(self.global_orient,dim=0).cpu().numpy()
        self.rotations = torch.stack(self.rotations,dim=0).cpu().numpy()
        print('\n')

        return self.root_translation,self.global_orient,self.rotations
    

    def save_obj(self, save_path):
        for batch_i in range(len(self.vertices)):
            result_i_path = save_path + '/result_{}/original_obj/'.format(batch_i) 
            os.makedirs(result_i_path)
            for frame_i in range(self.num_frames):
                frame_i_path = result_i_path + 'human_obj_{}.obj'.format(frame_i)
                vertices = self.vertices[batch_i][0, :, :, frame_i].squeeze().tolist()
                mesh = Trimesh(vertices=vertices,faces=self.faces)
                with open(frame_i_path, 'w') as fw:
                    mesh.export(fw, 'obj')