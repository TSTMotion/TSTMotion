import openai
import re
import os
import json
import numpy as np


SMPL_JOINT_NAMES = [
    "pelvis", # 0
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle", 
    "right_ankle",
    "spine3",
    "left_foot", # 10
    "right_foot", # 11
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand", # 20 left_wrist
    "right_hand", # 21 right_wrist
]


def rotate_3D_z(theta,x):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.matmul(x, R.transpose())


def get_overlap_area(a, b):
    dx = min(a[1], b[1]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[2], b[2])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0


class GPTError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return f"GPTError: {repr(self.value)}"
    

def display_plan(plan):
    str_plan = '{\n'

    for key, value in plan.items():
        if key == 'interaction_surface' or key == "feasible_interaction_direction":
            continue

        str_plan += '"{}":'.format(key)
        if type(value) == str:
            str_plan += '"{}",\n'.format(value)
        elif type(value) == list:
            str_plan += '['
            for temp in value:
                str_plan += str(temp) + ','
            str_plan += '],\n'
        else:
            str_plan += '{'
            for key_, value_ in value.items():
                if type(value_) == str:
                    str_plan += '"{}":"{}",'.format(key_,value_)
                else:
                    str_plan += '"{}":{},'.format(key_,value_)
            str_plan += '},\n'

    str_plan += '}'
    return str_plan


def check_plan(plan): # JSON
    plan_key = list(plan.keys())

    if 'motion' != plan_key[0]:
        raise GPTError("GPT Error")

    if 'motion_tendency' != plan_key[1]:
        raise GPTError("GPT Error")

    if 'motion_orientation' != plan_key[2]:
        raise GPTError("GPT Error")
    
    if plan['motion_orientation'] != 'forward' and plan['motion_orientation'] != 'backward':
        print("plan['motion_orientation'] != 'forward' and plan['motion_orientation'] != 'backward'")
        raise GPTError("GPT Error")
    
    if 'keyframe_1' != plan_key[3]:
        print("'keyframe_1' not in plan.keys()")
        raise GPTError("GPT Error")
    else:
        if 'pelvis' not in plan['keyframe_1'].keys():
            print("'pelvis' not in plan['keyframe_1'].keys()")
            raise GPTError("GPT Error")

    for key, value in plan.items():
        if 'keyframe' in key:
            for key_, value_ in value.items():
                if key_ in SMPL_JOINT_NAMES:
                    if type(value_) is not list:
                        print(value_)
                        print('value_ is not list')
                        raise GPTError("GPT Error")
                    for x in value_:
                        if type(x) is str:
                            print(x)
                            print('x is str')
                            raise GPTError("GPT Error") 

    return plan


# general for chatting witg GPT
def chat_completion(GPT_version,messages):
    
    response = openai.ChatCompletion.create(model=GPT_version,messages=messages,temperature=0.2)
    response_content = response.choices[0].message.content
    print('----LLM----')
    print(response_content,'\n')
    messages.append({"role": "assistant", "content": f"{response_content}"})

    return messages


def extract_from_messages(messages,result_path,name):
    target = messages[-1]['content']
    target = target.split('```')
    index = []
    for i in range(len(target)):
        if 'json' in target[i]:
            index.append(i)
    target = target[index[-1]]
    if 'Result:' in target:
        target = target.replace('Result:', "")[1:]
    if 'json' in target:
        target = target.replace("json", "")[1:]
    target = eval(target)
    if name != 'NO':
        target_path = os.path.join(result_path,'{}.json'.format(name))
        with open(target_path, 'w') as json_file:
            json.dump(target, json_file)
    return target


def extract_plan_from_messages(messages,result_path):
    plan = messages[-1]['content']
    plan = plan.split('```')
    index = []
    for i in range(len(plan)):
        if 'motion_orientation' in plan[i] and 'keyframe_' in plan[i]:
            index.append(i)
    plan = plan[index[-1]]
    plan = re.sub('//.*\n','\n',plan)
    if 'json' in plan:
        plan = plan.replace("json", "")[1:]
    if 'Result:' in plan:
        plan = plan.replace('Result:', "")[1:]
    plan = eval(plan)
    plan = check_plan(plan)

    plan_path = os.path.join(result_path,'plan.json')
    with open(plan_path, 'w') as json_file:
        json.dump(plan, json_file)

    return plan


def locate_target(args,caption,bounding_boxes,result_path, x_max_total, y_max_total):
    messages = []
    
    # Provide the GPT with background information 
    background_path = os.path.join(args.prompt_path,'locate_target.txt')
    with open(background_path, 'r') as file:
        background = file.read()
    messages.append({"role": "user", "content": f"{background}"})
    print('\n----USER----')
    print(background)


    response = 'I understand the task you\'re asking me to perform. Please offer the caption and bounding boxes.'
    print('\n----LLM----')
    print(response,'\n')
    messages.append({"role": "assistant", "content": f"{response}"})

    
    instruct = 'Now perform the specified task. If needed, you can make reasonable guesses.\n'
    instruct += '\nScene Scope: {' + 'x_min:0, x_max:{}, y_min:0, y_max:{}, z_min:0, z_max:250'.format(x_max_total, y_max_total) + '}\n'
    instruct += '\nBounding Boxes:\n'
    for box in bounding_boxes:
        instruct += '{'
        for key, value in box.items():
            if key!='box_number':
                instruct += f"{key}:{value},"
        instruct += '}\n'
    instruct += '\nCaption: ' + caption +'.\n'
    print('----USER----')
    print(instruct,'\n')
    messages.append({"role": "user", "content": f"{instruct}"})

    # messages = chat_completion(args.GPT_version,messages)
    # target = extract_target_from_messages(messages,result_path)
    loop_tag = True
    while loop_tag:
        try:
            messages = chat_completion(args.GPT_version,messages)
            target = extract_from_messages(messages,result_path,'locate_target')
            loop_tag = False
        except:
            if messages[-1]['role']!='user':
                messages.pop()
            pass

    return target


def get_top_K_index(data, k):
    data = np.array(data)
    idx = data.argsort()[-k:][::-1]
    return idx


def get_heightmap(points):
    ## For Robustness
    nums_pointcloud = points.shape[0]
    top_k_value = int(0.01*nums_pointcloud)
    top_k_x_min_index = get_top_K_index(-points[:,0],top_k_value)
    top_k_x_max_index = get_top_K_index(points[:,0],top_k_value)
    top_k_y_min_index = get_top_K_index(-points[:,1],top_k_value)
    top_k_y_max_index = get_top_K_index(points[:,1],top_k_value)
    top_k_z_min_index = get_top_K_index(-points[:,2],top_k_value)
    top_k_z_max_index = get_top_K_index(points[:,2],top_k_value)
    top_k_index = np.append(top_k_x_min_index, top_k_x_max_index)
    top_k_index = np.append(top_k_index, top_k_y_min_index)
    top_k_index = np.append(top_k_index, top_k_y_max_index)
    top_k_index = np.append(top_k_index, top_k_z_min_index)
    top_k_index = np.append(top_k_index, top_k_z_max_index)
    points = np.delete(points,top_k_index,axis=0)

    min_x, min_y, = np.min(points[:,0]), np.min(points[:,1])
    points[:,0] -= min_x
    points[:,1] -= min_y
    max_x, max_y, = np.max(points[:,0]), np.max(points[:,1])
    grids_ratio = 10
    grids_xnumber = np.int_(grids_ratio*max_x)
    grids_ynumber = np.int_(grids_ratio*max_y)
    grids_xrange = max_x / grids_xnumber
    grids_yrange = max_y / grids_ynumber
    target_girds = np.zeros((grids_xnumber,grids_ynumber)) 

    for i in range(len(points)):
        point = points[i]
        
        if point[0] == max_x:
            x_index = -1
        else:
            x_index = np.floor(point[0]/grids_xrange).astype(int)

        if point[1] == max_y:
            y_index = -1
        else:
            y_index = np.floor(point[1]/grids_yrange).astype(int)

        if point[2]> target_girds[x_index,y_index]:
            target_girds[x_index,y_index] = point[2]

    target_girds = target_girds*100
    target_girds = target_girds.astype(int)
    target_girds_f = target_girds.copy()
    target_girds = target_girds / 5
    target_girds = target_girds.astype(int)
    target_girds = target_girds * 5
    target_girds = target_girds.astype(int)

    heightmap = ''
    heightmap += '[\n'
    for i in range(grids_xnumber):
        heightmap += '['
        for j in range(grids_ynumber):
            if j == grids_ynumber-1:
                heightmap += '{}'.format(target_girds[i,j])
            else:
                heightmap += '{}, '.format(target_girds[i,j])
        heightmap += '],\n'
    heightmap += ']\n\n'

    return heightmap, target_girds_f, min_x, min_y, grids_xrange, grids_yrange


def grid_to_coord(predicted_results, target_grids, min_x, min_y, grids_xrange, grids_yrange):
    interaction_grid = list(predicted_results['interaction_grid'])
    target_coord = [0,0,0]
    target_coord[2] = target_grids[interaction_grid[0],interaction_grid[1]]

    target_coord[0] = min_x + grids_xrange * (interaction_grid[0]+0.5)
    target_coord[1] = min_y + grids_yrange * (interaction_grid[1]+0.5)

    target_coord[0] = target_coord[0]*100
    target_coord[0] = target_coord[0].astype(int)
    target_coord[1] = target_coord[1]*100
    target_coord[1] = target_coord[1].astype(int)

    return target_coord


def top_scene(args,coord,segment,caption,bounding_boxes,result_path,target):
    messages = []
    
    # Provide the GPT with background information 
    background_path = os.path.join(args.prompt_path,'top_scene.txt')
    with open(background_path, 'r') as file:
        background = file.read()
    messages.append({"role": "user", "content": f"{background}"})
    print('\n----USER----')
    print(background)
    
    response = 'I understand the task you\'re asking me to perform. Please provide the road map and the caption.'
    print('\n----LLM----')
    print(response,'\n')
    messages.append({"role": "assistant", "content": f"{response}"})


    instruct = 'Now perform the specified task. If needed, you can make reasonable guesses.\n'
    roadmap = get_roadmap(coord,segment,bounding_boxes,target)
    instruct += '\nRoad Map of the Scene:\n'
    instruct += roadmap

    instruct += 'Caption: ' + caption +'.'
    print('----USER----')
    print(instruct,'\n')
    messages.append({"role": "user", "content": f"{instruct}"})

    loop_tag = True
    while loop_tag:
        try:
            messages = chat_completion(args.GPT_version,messages)
            predicted_results = extract_from_messages(messages,result_path,'top_scene')
            loop_tag = False
        except:
            if messages[-1]['role']!='user':
                messages.pop()
            pass

    target['feasible_interaction_direction'] = predicted_results['feasible_interaction_direction']
    return target

def top_target(args,coord,segment,caption,bounding_boxes,result_path,target):
    messages = []
    
    # Provide the GPT with background information 
    background_path = os.path.join(args.prompt_path,'top_target.txt')
    with open(background_path, 'r') as file:
        background = file.read()
    messages.append({"role": "user", "content": f"{background}"})
    print('\n----USER----')
    print(background)


    response = 'I understand the task you\'re asking me to perform. Please provide the height map and the caption.'
    print('\n----LLM----')
    print(response,'\n')
    messages.append({"role": "assistant", "content": f"{response}"})


    instruct = 'Now perform the specified task. If needed, you can make reasonable guesses.\n'
    # instruct += '\nScene Scope: {' + 'x_min:0, x_max:{}, y_min:0, y_max:{}, z_min:0, z_max:250'.format(x_max_scene, y_max_scene) + '}\n'
    for box in bounding_boxes:
        if box['midpoint'] == target['target']['midpoint'] and box['label'] == target['target']['label']:
            index = box['box_number']
            break
    target_pointcloud = coord[segment[:,index]]
    # x_min_target, y_min_target, z_min_target,  = np.int_(np.min(target_pointcloud[:,0]*100)), np.int_(np.min(target_pointcloud[:,1]*100)), np.int_(np.min(target_pointcloud[:,2]*100))
    # x_max_target, y_max_target, z_max_target,  = np.int_(np.max(target_pointcloud[:,0]*100)), np.int_(np.max(target_pointcloud[:,1]*100)), np.int_(np.max(target_pointcloud[:,2]*100))
    # print(x_min_target, y_min_target, z_min_target)
    # print(x_max_target, y_max_target, z_max_target)
    # instruct += '\nTarget Scope: {' + 'x_min:{}, x_max:{}, y_min:{}, y_max:{}, z_min:{}, z_max:{}'.format(
    #     x_min_target, x_max_target,y_min_target, y_max_target,z_min_target, z_max_target
    # ) + '}\n'

    heightmap, target_grids, min_x, min_y, grids_xrange, grids_yrange = get_heightmap(target_pointcloud)
    instruct += '\nHeight Map of Target:\n'
    instruct += heightmap

    instruct += '\nFeasible Interaction Direction:\n'
    feasible_interaction_direction = "["
    for i in target["feasible_interaction_direction"]:
        feasible_interaction_direction += "\""+ i + '",'
    feasible_interaction_direction += ']\n'
    instruct += feasible_interaction_direction

    instruct += 'Caption: ' + caption +'.'
    print('----USER----')
    print(instruct,'\n')
    messages.append({"role": "user", "content": f"{instruct}"})

    # messages = chat_completion(args.GPT_version,messages)
    # target_grids = extract_from_messages(messages,result_path, 'target_grids')
    loop_tag = True
    while loop_tag:
        try:
            messages = chat_completion(args.GPT_version,messages)
            predicted_results = extract_from_messages(messages,result_path,'top_target')
            loop_tag = False
        except:
            if messages[-1]['role']!='user':
                messages.pop()
            pass

    target['interaction_joint'] = predicted_results['interaction_joint']
    target['interaction_position'] = grid_to_coord(predicted_results, target_grids, min_x, min_y, grids_xrange, grids_yrange)
    interaction_direction = predicted_results['interaction_direction']
    if interaction_direction == 'top':
        interaction_direction = 'x_min'
    elif interaction_direction == 'bottom':
        interaction_direction = 'x_max'
    elif interaction_direction == 'left':
        interaction_direction = 'y_min'
    elif interaction_direction == 'right':
        interaction_direction = 'y_max'
    elif interaction_direction == 'top-left':
        interaction_direction = 'x_min-y_min'
    elif interaction_direction == 'top-right':
        interaction_direction = 'x_min-y_max'
    elif interaction_direction == 'bottom-left':
        interaction_direction = 'x_max-y_min'
    elif interaction_direction == 'bottom-right':
        interaction_direction = 'x_max-y_max'
    target['interaction_direction'] = interaction_direction
    return target


def top_plan(args,caption,result_path,target):
    messages = []
    
    # Provide the GPT with background information 
    background_path = os.path.join(args.prompt_path,'top_plan.txt')
    with open(background_path, 'r') as file:
        background = file.read()
    messages.append({"role": "user", "content": f"{background}"})
    print('\n----USER----')
    print(background)

    response = 'I understand the task you\'re asking me to perform. Please provide the target and the caption.'
    print('\n----LLM----')
    print(response,'\n')
    messages.append({"role": "assistant", "content": f"{response}"})

    instruct = 'Now perform the specified task. If needed, you can make reasonable guesses.\n'
    instruct += '\nTarget:\n'
    instruct += display_plan(target)

    instruct += '\n\nCaption: ' + caption +'.\n'
    print('----USER----')
    print(instruct,'\n')
    messages.append({"role": "user", "content": f"{instruct}"})

    # messages = chat_completion(args.GPT_version,messages)
    # target = extract_target_from_messages(messages,result_path)
    loop_tag = True
    while loop_tag:
        try:
            messages = chat_completion(args.GPT_version,messages)
            plan = extract_plan_from_messages(messages,result_path)
            loop_tag = False
        except:
            if messages[-1]['role']!='user':
                messages.pop()
            pass

    return plan


def get_roadmap(coord,segment,bounding_boxes,target):
    grids_ratio = 100
    max_scene_x, max_scene_y = np.max(coord[:,0]), np.max(coord[:,1])
    grids_xnumber = np.int_(grids_ratio*max_scene_x)
    grids_ynumber = np.int_(grids_ratio*max_scene_y)
    grids_xrange = max_scene_x / grids_xnumber
    grids_yrange = max_scene_y / grids_ynumber
    map = np.zeros((grids_xnumber,grids_ynumber))


    for box in bounding_boxes:
        # print(box)
        if box['midpoint'] == target['target']['midpoint'] and box['label'] == target['target']['label']:
            index = box['box_number']

        min_x, min_y = box['x_min']/100, box['y_min']/100
        max_x, max_y = box['x_max']/100, box['y_max']/100
        for j in range(grids_xnumber):
            for k in range(grids_ynumber):
                matrix_1 = [j * grids_xrange, (j+1) * grids_xrange, k * grids_xrange, (k+1) * grids_xrange]
                matrix_2 = [min_x, max_x, min_y, max_y]
                overlap_area = get_overlap_area(matrix_1, matrix_2)
                if overlap_area > 0.5 * grids_xrange * grids_yrange:
                    # map[j][k] = box['box_number']
                    map[j][k] = 1

    grids_ratio = 10
    grids_xnumber = np.int_(grids_ratio*max_scene_x)
    grids_ynumber = np.int_(grids_ratio*max_scene_y)
    grids_xrange = max_scene_x / grids_xnumber
    grids_yrange = max_scene_y / grids_ynumber
    roadmap = np.zeros((grids_xnumber,grids_ynumber))

    for i in range(grids_xnumber):
        for j in range(grids_ynumber):
            if i != (grids_xnumber-1):
                x_min_index = i * 10
                x_max_index = i * 10 + 10
            else :
                x_min_index = i * 10
                x_max_index = map.shape[0]-1

            if j != (grids_ynumber-1):
                y_min_index = j * 10
                y_max_index = j * 10 + 10
            else:
                y_min_index = j * 10
                y_max_index = map.shape[1]-1

            grids_sum = np.sum(map[x_min_index:x_max_index,y_min_index:y_max_index])
            if grids_sum < 0.5*(x_max_index-x_min_index)*(y_max_index-y_min_index):
                roadmap[i][j] = 0
            else:
                roadmap[i][j] = 1

            if i==0 or j ==0 or i == grids_xnumber-1 or j == grids_ynumber-1:
                roadmap[i][j] = 1


    target_pc = coord[segment[:,index]]
    min_x, min_y = np.min(target_pc[:,0]), np.min(target_pc[:,1])
    max_x, max_y = np.max(target_pc[:,0]), np.max(target_pc[:,1])
    for j in range(grids_xnumber):
        for k in range(grids_ynumber):
            matrix_1 = [j * grids_xrange, (j+1) * grids_xrange, k * grids_xrange, (k+1) * grids_xrange]
            matrix_2 = [min_x, max_x, min_y, max_y]
            overlap_area = get_overlap_area(matrix_1, matrix_2)
            if overlap_area > 0.5 * grids_xrange * grids_yrange:
                roadmap[j][k] = 2

    roadmap_str = ''
    roadmap_str += '[\n'
    for i in range(grids_xnumber):
        roadmap_str += '['
        for j in range(grids_ynumber):
            if j == grids_ynumber-1:
                roadmap_str += '{}'.format(roadmap[i,j].astype(np.int16))
            else:
                roadmap_str += '{}, '.format(roadmap[i,j].astype(np.int16))
        roadmap_str += '],\n'
    roadmap_str += ']\n\n'

    return roadmap_str


def coarse_target(args,coord,segment,caption,bounding_boxes,result_path,target):
    messages = []
    
    # Provide the GPT with background information 
    background_path = os.path.join(args.prompt_path,'coarse_target.txt')
    with open(background_path, 'r') as file:
        background = file.read()
    messages.append({"role": "user", "content": f"{background}"})
    print('\n----USER----')
    print(background)
    
    response = 'I understand the task you\'re asking me to perform. Please provide the road map and the caption.'
    print('\n----LLM----')
    print(response,'\n')
    messages.append({"role": "assistant", "content": f"{response}"})


    instruct = 'Now perform the specified task. If needed, you can make reasonable guesses.\n'
    roadmap = get_roadmap(coord,segment,bounding_boxes,target)
    instruct += '\nRoad Map of the Scene:\n'
    instruct += roadmap

    instruct += 'Caption: ' + caption +'.'
    print('----USER----')
    print(instruct,'\n')
    messages.append({"role": "user", "content": f"{instruct}"})

    loop_tag = True
    while loop_tag:
        try:
            messages = chat_completion(args.GPT_version,messages)
            predicted_results = extract_from_messages(messages,result_path,'coarse_target')
            loop_tag = False
        except:
            if messages[-1]['role']!='user':
                messages.pop()
            pass

    motion_start = predicted_results["motion_start"]
    motion_start[0] = np.int_((0.5 + motion_start[0]) * 10)
    motion_start[1] = np.int_((0.5 + motion_start[1]) * 10)

    motion_end = predicted_results["motion_end"]
    motion_end[0] = np.int_((0.5 + motion_end[0]) * 10)
    motion_end[1] = np.int_((0.5 + motion_end[1]) * 10)

    target["motion_start"], target["motion_end"] = motion_start, motion_end
    target['interaction_joint'] = predicted_results['interaction_joint']

    return target


def coarse_plan(args,caption,result_path,target,x_max_total, y_max_total):
    messages = []
    
    # Provide the GPT with background information 
    background_path = os.path.join(args.prompt_path,'coarse_plan.txt')
    with open(background_path, 'r') as file:
        background = file.read()
    messages.append({"role": "user", "content": f"{background}"})
    print('\n----USER----')
    print(background)

    response = 'I understand the task you\'re asking me to perform. Please provide the target and the caption.'
    print('\n----LLM----')
    print(response,'\n')
    messages.append({"role": "assistant", "content": f"{response}"})

    instruct = 'Now perform the specified task. If needed, you can make reasonable guesses.\n'
    instruct += '\nScene Scope: {' + 'x_min:0, x_max:{}, y_min:0, y_max:{}, z_min:0, z_max:250'.format(x_max_total, y_max_total) + '}\n'
    instruct += '\nTarget:\n'
    instruct += display_plan(target)

    instruct += '\n\nCaption: ' + caption +'.\n'
    print('----USER----')
    print(instruct,'\n')
    messages.append({"role": "user", "content": f"{instruct}"})

    # messages = chat_completion(args.GPT_version,messages)
    # target = extract_target_from_messages(messages,result_path)
    loop_tag = True
    while loop_tag:
        try:
            messages = chat_completion(args.GPT_version,messages)
            plan = extract_plan_from_messages(messages,result_path)
            loop_tag = False
        except:
            if messages[-1]['role']!='user':
                messages.pop()
            pass

    return plan

# To get the complete plan of human skeleton moition from GPT
def get_gpt_plan(args,coord,segment,caption,bounding_boxes,result_path):
    if os.path.exists(os.path.join(result_path,'plan.json')):
        with open(os.path.join(result_path,'plan.json'), 'r') as file:
            plan = json.load(file)
        return None, plan

    x_max_total, y_max_total = np.int_(np.max(coord[:,0])*100),np.int_(np.max(coord[:,1])*100)
    target = locate_target(args,caption,bounding_boxes,result_path,x_max_total,y_max_total)
    
    if target['interaction_surface'] == 'top':
        target = top_scene(args,coord,segment,caption,bounding_boxes,result_path,target)
        target = top_target(args,coord,segment,caption,bounding_boxes,result_path,target)
        plan = top_plan(args,caption,result_path,target)
        return target, plan['motion'], plan
    elif target['interaction_surface'] == 'coarse':
        target = coarse_target(args,coord,segment,caption,bounding_boxes,result_path,target)
        plan = coarse_plan(args,caption,result_path,target,x_max_total, y_max_total )
        return target, plan['motion'], plan



def get_keyframes(plan):
    keyframes = []
    for key,value in plan.items():
        if 'keyframe' in key:
            keyframe = int(key.replace('keyframe_','')) - 1
            keyframes.append(keyframe)
    return keyframes


def get_keyjoints(plan):
    keyjoints = []
    keyjoints_loc = []

    for key_,frame_ in plan.items():
        if 'keyframe' in key_:
            keyjoint = []
            keyjoint_loc = []
            for key, value in frame_.items():
                if key in SMPL_JOINT_NAMES:
                    keyjoint.append(SMPL_JOINT_NAMES.index(key))
                    if type(value)==str:
                        value = eval(value)
                    keyjoint_loc.append(value)

            keyjoints.append(keyjoint)
            keyjoints_loc.append(keyjoint_loc)

    return keyjoints,keyjoints_loc


def get_check_plan(args,all_motions,caption,coord,target,plan,keyframes,result_path):
    x_max_total, y_max_total = np.int_(np.max(coord[:,0])*100),np.int_(np.max(coord[:,1])*100)
    messages = []

    if target['interaction_surface'] == "top":
        background_path = os.path.join(args.prompt_path,'top_postcheck.txt')
        with open(background_path, 'r') as file:
            background = file.read()
    elif target['interaction_surface'] == "coarse":
        background_path = os.path.join(args.prompt_path,'coarse_postcheck.txt')
        with open(background_path, 'r') as file:
            background = file.read()

    background += '\n\nScene Scope: {' + 'x_min:0, x_max:{}, y_min:0, y_max:{}, z_min:0, z_max:250'.format(x_max_total, y_max_total) + '}\n'
    background += '\nTarget:\n'
    background += display_plan(target)
    background += '\n\nCaption: ' + caption +'.\n'
    background += '\nPlan:\n'
    background += '```json\n'
    background += display_plan(plan)
    background += '\n```'


    print('\n----USER----')
    interaction_joint = target['interaction_joint']
    show_joints = [0]
    for i in range(len(SMPL_JOINT_NAMES)):
        if interaction_joint == SMPL_JOINT_NAMES[i] and i !=0:
            show_joints.append(i)


    instruct = '\n\nHere we have {} results.'.format(len(all_motions))
    for i in range(len(all_motions)):
        instruct_i = '\n\nThis is the {}-th result:\n'.format(i+1)
        dict_i = {}

        for j in keyframes:
            index_j = 'keyframe_{}'.format(j+1)
            dict_i[index_j] = {}

            for k in show_joints:
                index_k = SMPL_JOINT_NAMES[k]
                temp_motion = all_motions[i,k,:,j].astype(np.int16)
                dict_i[index_j][index_k] = temp_motion.tolist()

        dict_i = display_plan(dict_i)
        instruct_i = instruct_i + dict_i + '\n'
        instruct += instruct_i

    instruct += '\n'
    instruct += 'Now, you should generate a new "plan" in the same format. If needed, you can make reasonable guess.'
    print(background+instruct,'\n')
    messages.append({"role": "user", "content": f"{background+instruct}"})
    loop_tag = True
    while loop_tag:
        try:
            messages = chat_completion(args.GPT_version,messages)
            check_plan = extract_plan_from_messages(messages,result_path)
            loop_tag = False
        except:
            if messages[-1]['role']!='user':
                messages.pop()
            pass
    return check_plan