import copy
import numpy as np
import torch


def rotate_2D(theta,x):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    return np.dot(R,x)


def rotate_2D_z(theta,x):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    return np.matmul(x,R.transpose())


def rotate_3D_z(theta,x):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.matmul(x, R.transpose())


def rotate_scene(scene):
    mean_x, mean_y = np.mean(scene[:,0]), np.mean(scene[:,1])
    max_area = 99999999999
    theta_scene  = 0

    for theta in range(360):
        rot_scene = scene.copy()
        rot_scene[:,0] -= mean_x
        rot_scene[:,1] -= mean_y
        rot_scene= rotate_3D_z(theta/180*np.pi,rot_scene)

        min_x, min_y = np.min(rot_scene[:,0]), np.min(rot_scene[:,1])
        max_x, max_y = np.max(rot_scene[:,0]), np.max(rot_scene[:,1])
        curr_area = (max_x-min_x) * (max_y-min_y)
        if curr_area < max_area:
            max_area = curr_area
            theta_scene = theta

    rot_scene = scene.copy()
    rot_scene[:,0] -= mean_x
    rot_scene[:,1] -= mean_y
    rot_scene = rotate_3D_z(theta_scene/180*np.pi,rot_scene)
    min_x, min_y, min_z = np.min(rot_scene[:,0]), np.min(rot_scene[:,1]), np.min(rot_scene[:,2])
    rot_scene[:,0] -= min_x
    rot_scene[:,1] -= min_y
    rot_scene[:,2] -= min_z

    return rot_scene, mean_x, mean_y, theta_scene/180*np.pi, min_x, min_y, min_z


def scene_to_GPT(bounding_boxes):
    for box in bounding_boxes:
        box['x_min'] = np.int_(box['x_min']*100)
        box['y_min'] = np.int_(box['y_min']*100)
        box['z_min'] = np.int_(box['z_min']*100)
        box['x_max'] = np.int_(box['x_max']*100)
        box['y_max'] = np.int_(box['y_max']*100)
        box['z_max'] = np.int_(box['z_max']*100)
        for i in range(len(box['midpoint'])):
            box['midpoint'][i] = np.int_(box['midpoint'][i]*100)

    return bounding_boxes


def GPT_to_MDM(plan):
    plan_in_MDM = copy.deepcopy(plan)
    pelvis_coord = list(plan_in_MDM.values())
    direction = plan['motion_orientation']
    if direction == 'forward':
        direction = 1
    elif direction == 'backward':
        direction = -1

    starting = np.copy(pelvis_coord[3]['pelvis'])
    ending = np.copy(pelvis_coord[-1]['pelvis'])
    forward = (ending - starting) * direction

    # if caption[0:7] == 'looking' or caption[0:6] == 'facing':
    #     if ',' in caption:
    #         caption = caption.split(',')[1][1:]
    # caption = caption.split(' ')
    # if caption[0] == 'walk' or caption[0] == 'stand':
    #     forward = ending - starting
    # elif caption[0] == 'lie' or caption[0] == 'sit':
    #     forward = starting - ending
    # else:
    #     forward = ending - starting
    
    if forward[0]==0 and forward[1] == 0:
        theta = -np.pi/2
    elif forward[0]==0 and forward[1] != 0:
        theta = np.pi/2 * np.sign(forward[1])
    elif forward[0]<=0 and forward[1] == 0:
        theta = np.pi
    elif forward[0]>=0 and forward[1] == 0:
        theta = 0
    elif forward[0]<0 and forward[1]>=0:
        theta = np.arctan(forward[1]/forward[0])
        theta = np.pi+theta
    elif forward[0]<0 and forward[1]<0:
        theta = np.arctan(forward[1]/forward[0])
        theta = np.pi+theta
    else:
        theta = np.arctan(forward[1]/forward[0])
    # print('>>> theta:',theta)

    for key,value in plan.items():
        if 'keyframe' in key:
            for key_,value_ in value.items():
                if key_ != 'state':
                    value_ -= starting
                    value_[:2] = rotate_2D(-theta-np.pi/2,value_[:2])
                    value_[1:] = rotate_2D(-np.pi/2,value_[1:])
                    value_[1] += starting[2]
                    value_ = value_ / 100
                    plan_in_MDM[key][key_] = value_

    return theta, plan_in_MDM, starting, ending


def GPT_to_scene(starting, ending):
    starting = starting /100
    ending = ending /100
    return starting, ending


def MDM_to_GPT(motion_tendency,all_motions,starting,ending,theta):
    joint,frame = all_motions.shape[1],all_motions.shape[3]
    for i in range(len(all_motions)):
        motion = all_motions[i] # (joint(22), 3, frame)
        motion = motion.transpose((1,0,2))  # (3, joint(22), frame)
        origin = motion[:,0,0].copy()
        origin = origin[:,None,None]
        motion = motion - origin
        motion = motion.reshape(3,-1) # (3, joint(22)*frame)

        motion[1:,:] = rotate_2D(np.pi/2,motion[1:,:])
        motion[:2,:] = rotate_2D(theta+np.pi/2,motion[:2,:])
        motion = motion.reshape(3,joint,frame) # (3, joint(22),frame)
        motion = motion * 100

        if motion_tendency =='toward':
            last = motion[:,0,-1].copy()
            motion += ending[:,None,None] - last[:,None,None]

        elif motion_tendency =='away':
            first = motion[:,0,0].copy()
            motion += starting[:,None,None] - first[:,None,None]

        motion = motion.transpose((1,0,2))  # (joint(22), 3, frame)
        all_motions[i] = motion

    return all_motions