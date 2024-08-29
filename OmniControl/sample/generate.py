# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import numpy as np
import torch
from utils import dist_util
from data_loaders.humanml.scripts.motion_process import recover_from_ric


def keyjoints_to_hint(keyframes,keyjoints,keyjoints_loc,max_frames = 196):   # [0, 39, 79] [[0], [0], [0]] [[array([0.  , 0.85, 0.  ])], [array([0.  , 0.85, 1.28])], [array([0.  , 0.85, 2.56])]]
    raw_mean = np.load('./OmniControl/dataset/humanml_spatial_norm/Mean_raw.npy').reshape(22, 1, 3)
    raw_std = np.load('./OmniControl/dataset/humanml_spatial_norm/Std_raw.npy').reshape(22, 1, 3)
     
    hint = torch.zeros((1,max_frames,22,3))
    for i in range(len(keyframes)):
        for j in range(len(keyjoints[i])):
            hint_value = (torch.tensor(keyjoints_loc[i][j]) - raw_mean[keyjoints[i][j]]) / raw_std[keyjoints[i][j]]
            hint[0,keyframes[i],keyjoints[i][j],:] = hint_value
    hint = hint.reshape((hint.shape[0], max_frames, -1))
    return hint


def generate_motion(args,data,model,diffusion,caption,keyframes,keyjoints,keyjoints_loc):
    dist_util.setup_dist(args.device)

    max_frames = 196 
    model_kwargs ={'y':{}}
    model_kwargs['y']['mask'] = torch.ones((args.num_samples,1,1,max_frames))
    model_kwargs['y']['lengths'] = torch.ones((args.num_samples)) * max_frames
    model_kwargs['y']['text'] = [caption] * args.num_samples
    model_kwargs['y']['tokens'] = [None] * args.num_samples
    model_kwargs['y']['hint'] = keyjoints_to_hint(keyframes,keyjoints,keyjoints_loc,max_frames)
    model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())

    # add CFG scale to batch
    print(f'>>> Start sampling')
    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, max_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        scale_lambda = args.scale_lambda,
        scale_eta = args.scale_eta,
    )


    # Recover XYZ *positions* from HumanML3D vector representation
    n_joints = 22 if sample.shape[1] == 263 else 21
    sample = data.dataset.t2m_dataset.inv_transform(sample.detach().cpu().permute(0, 2, 3, 1)).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)


    all_motions = []
    all_motions.append(sample.cpu().numpy())
    all_motions = np.concatenate(all_motions, axis=0)
    total_num_samples = args.num_samples * args.num_repetitions
    if keyframes[-1]+1 < max_frames:
        all_motions = all_motions[:total_num_samples,:,:,:keyframes[-1]+1]  # [bs, njoints, 6, seqlen]
    else:
        all_motions = all_motions[:total_num_samples,:,:,:]  # [bs, njoints, 6, seqlen]
    return all_motions
