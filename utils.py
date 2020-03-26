from collections import deque
import random
import os
import math

import torch
import numpy as np
import torch.nn as nn
import gym
import daisy_kinematics
import daisy_raibert_controller

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def normalization(a, mean, std):
    return (a-mean)/std
    
def inverse_normalization(a, mean, std):
    return a*std + mean
    
class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim,delta_obs_dim, device, capacity,sim= True, save_dir = None):
        self.device = device
        self.capacity = capacity

        if type(obs_dim) == int:
            self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
            self.next_obses = np.empty((capacity, delta_obs_dim), dtype=np.float32)
        else:
            self.obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
            self.next_obses = np.empty((capacity, *obs_dim), dtype=np.uint8)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.all_mean_var = np.array([
            np.zeros(obs_dim),
            np.ones(obs_dim),
            np.zeros(action_dim),
            np.ones(action_dim),
            np.zeros(delta_obs_dim),
            np.ones(delta_obs_dim),
        ])

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save_buffer(self):
        save_dir = make_dir(os.path.join('./buffer_data'))
        np.save(save_dir+'/obses', self.obses)
        np.save(save_dir+'/actions', self.actions)
        np.save(save_dir+'/rewards', self.rewards)
        np.save(save_dir+'/next_obses', self.next_obses)
        np.save(save_dir+'/not_dones', self.not_dones)
        np.save(save_dir+'/idx',np.array([self.idx]))

        # save mean and var
        if self.full:
            self.all_mean_var[0]= np.mean(self.obses, axis = 0)
            self.all_mean_var[1]= np.std(self.obses, axis = 0)
            self.all_mean_var[2]= np.mean(self.actions, axis = 0)
            self.all_mean_var[3]= np.std(self.actions, axis = 0)
            self.all_mean_var[4]= np.mean(self.next_obses, axis = 0)
            self.all_mean_var[5]= np.std(self.next_obses, axis = 0)
        else:
            self.all_mean_var[0]= np.mean(self.obses[0:self.idx], axis = 0)
            self.all_mean_var[1]= np.std(self.obses[0:self.idx], axis = 0)
            self.all_mean_var[2]= np.mean(self.actions[0:self.idx], axis = 0)
            self.all_mean_var[3]= np.std(self.actions[0:self.idx], axis = 0)
            self.all_mean_var[4]= np.mean(self.next_obses[0:self.idx], axis = 0)
            self.all_mean_var[5]= np.std(self.next_obses[0:self.idx], axis = 0)
        np.save(save_dir+'/all_mean_var', self.all_mean_var)

    def load_mean_var(self):
        '''
        0: mean of obses; 1: var of obses; 2: mean of actions; 3: var of actions; 4: mean of next_obses; 5: var of next_obses
        '''
        self.all_mean_var = np.load('./buffer_data/all_mean_var.npy')
        return self.all_mean_var

    def load_buffer(self,):
        save_dir = './buffer_data'
        self.obses = np.load(save_dir+'/obses.npy')
        self.actions = np.load(save_dir+'/actions.npy')
        self.rewards = np.load(save_dir+'/rewards.npy')
        self.next_obses = np.load(save_dir+'/next_obses.npy')
        self.not_dones = np.load(save_dir+'/not_dones.npy')
        self.idx = np.load(save_dir+'/idx.npy')[0]

        return self.load_mean_var()
        


class CoM_frame_MPC():
    def HL_obs(self, state):
        '''
        Extract useful information from state(dict) and form to np.array()
        input:
            state: dict
        output:
            HL_obs: np.array
                    now the obs includes: com velocity in xyz, yaw information
        '''
        # TODO: form the HL_obs & translate com velocity to com frame
        high_level_obs = []
        world2com_ori = daisy_kinematics.World2Com_ori(state['base_ori_euler'])
        vel2com = daisy_kinematics.coordinate_trans(state['base_velocity'], world2com_ori)
        return vel2com[0:2]

    def HL_delta_obs(self, pre_com_state,post_com_state): 
        '''
        Extract useful information from state(dict) and form to np.array()
        input:
            state: dict
        output:
            HL_obs: np.array
        '''
        high_level_delta_obs = []
        delta_yaw = post_com_state['base_ori_euler'][2] - pre_com_state['base_ori_euler'][2]
        if delta_yaw > np.pi:
            delta_yaw = 2 * np.pi - delta_yaw
        elif delta_yaw< -np.pi:
            delta_yaw = - 2 * np.pi - delta_yaw

        # print(delta_yaw)

        world2com_ori = daisy_kinematics.World2Com_ori(pre_com_state['base_ori_euler'])
        delta_pos = []
        delta_pos.append(post_com_state['base_pos_x'][0] - pre_com_state['base_pos_x'][0])
        delta_pos.append(post_com_state['base_pos_y'][0] - pre_com_state['base_pos_y'][0])
        delta_pos.append(0)
        delta_pos_com = daisy_kinematics.coordinate_trans(np.array(delta_pos), world2com_ori)
        delta_vel_com = daisy_kinematics.coordinate_trans(post_com_state['base_velocity'] - pre_com_state['base_velocity'] , world2com_ori)

        high_level_delta_obs.append(delta_yaw)
        for i in range(2):
            high_level_delta_obs.append(delta_pos_com[i])
        for i in range(2):
            high_level_delta_obs.append(delta_vel_com[i])

        return np.array(high_level_delta_obs)

    def run_mpc_calc_cost(self, HL_obs_buffer, model, target, latent_action_sample, world_buffer, cost, mean_var):
        latent_action_sample_all = np.swapaxes(latent_action_sample,0,1)
        last_velocity = HL_obs_buffer[0][:]
        for i in range(np.shape(latent_action_sample)[0]):
            latent_action_norm = normalization(latent_action_sample[i],mean_var[2], mean_var[3] )
            HL_obs_buffer_norm = normalization(HL_obs_buffer, mean_var[0], mean_var[1])
            predict_delta_state_norm = model.predict_para(HL_obs_buffer_norm, latent_action_norm)
            predict_delta_state = inverse_normalization(predict_delta_state_norm,  mean_var[4], mean_var[5])
            HL_obs_buffer, world_buffer = self.calc_next_input(predict_delta_state, world_buffer)
            cost = calc_cost(cost, predict_delta_state, world_buffer, target,latent_action_sample_all, last_velocity)
        return cost

    def calc_next_input(self, predict_delta_state, world_buffer):
        predict_size = np.shape(predict_delta_state)
        HL_obs_buffer = np.empty((predict_size[0], predict_size[1]-3))
        for i in range(predict_size[0]):
            com_ori = np.array([0, 0 , world_buffer[i][0]])
            delta_ori_com = predict_delta_state[i][0]
            delta_pos_com = predict_delta_state[i][1:3] # delta x,y in com frame
            delta_vel_com = predict_delta_state[i][3:5] # delta vx, vy, in com frame

            com2world_ori = daisy_kinematics.CoM2World_ori(com_ori)
            delta_pos_world = daisy_kinematics.coordinate_trans(np.append(delta_pos_com,0), com2world_ori)
            delta_vel_world = daisy_kinematics.coordinate_trans(np.append(delta_vel_com,0), com2world_ori)

            world_buffer[i][0] = world_buffer[i][0] + delta_ori_com
            world_buffer[i][1:3] = world_buffer[i][1:3] + delta_pos_world[0:2]
            world_buffer[i][3:5] = world_buffer[i][3:5] + delta_vel_world[0:2]
            
            world2com_ori = daisy_kinematics.World2Com_ori(np.array([0,0,world_buffer[i][0]]))
            HL_obs_buffer[i] = daisy_kinematics.coordinate_trans(np.append(world_buffer[i][3:5],0), world2com_ori)[0:2]
        return HL_obs_buffer, world_buffer
    
    def predict2world(self, pre_com_state, predict_delta_state):
        delta_ori_com = predict_delta_state[0]
        delta_pos_com = predict_delta_state[1:3] # delta x,y in com frame
        delta_vel_com = predict_delta_state[3:5]
        com2world_ori = daisy_kinematics.CoM2World_ori(np.array([0,0,pre_com_state['base_ori_euler'][2]]))
        delta_pos_world = daisy_kinematics.coordinate_trans(np.append(delta_pos_com,0), com2world_ori)
        delta_vel_world = daisy_kinematics.coordinate_trans(np.append(delta_vel_com,0), com2world_ori)
        state_world = []
        state_world.append(delta_ori_com+pre_com_state['base_ori_euler'][2])
        state_world.append(delta_pos_world[0] + pre_com_state['base_pos_x'][0])
        state_world.append(delta_pos_world[1] + pre_com_state['base_pos_y'][0])
        state_world.append(delta_vel_world[0] + pre_com_state['base_velocity'][0])
        state_world.append(delta_vel_world[1] + pre_com_state['base_velocity'][1])
        return state_world

 
class CoM_frame_RL():
    def HL_obs(self, state, target):
        high_level_obs = []
        
        yaw_term = target[2] - state['base_ori_euler'][2]

        world2com_ori = daisy_kinematics.World2Com_ori(state['base_ori_euler'])
        pos = np.array([target[0] - state['base_pos_x'][0], target[1] - state['base_pos_y'][0],0])
        pos2com = daisy_kinematics.coordinate_trans(pos, world2com_ori) # position term

        vel2com = daisy_kinematics.coordinate_trans(state['base_velocity'], world2com_ori) # velocity term

        high_level_obs.append(yaw_term)
        for i in range(2):
            high_level_obs.append(pos2com[i])
        for i in range(2):
            high_level_obs.append(vel2com[i])
        
        return np.array(high_level_obs)

    
    def calc_reward(self,state,target):  
        cost = (2 * np.linalg.norm(np.array([target[0] - state['base_pos_x'][0], target[1] - state['base_pos_y'][0]])) 
                            + 1*np.linalg.norm(target[2] - state['base_ori_euler'][2]))
        return -cost


def check_robot_dead(state):
    '''
    check if the robot falls
    input:
        state: dict
    output:
        bool
    '''
    #TODO: define termination condition
    if np.abs(state['base_ori_euler'][0])>0.5 or np.abs(state['base_ori_euler'][1])>0.3:
        return True
    return False

def check_data_useful():
    return True

def calc_next_state(state,predict_delta_state):
    next_state = np.zeros(np.shape(predict_delta_state)[0]-1)
    next_state[0] = math.atan2(predict_delta_state[0], predict_delta_state[1])
    next_state[1] = predict_delta_state[2] + state['base_pos_x'][0]
    next_state[2] = predict_delta_state[3] + state['base_pos_y'][0]
    next_state[3:] =  np.copy(predict_delta_state[4:])

    high_level_obs= []
    high_level_obs.append(predict_delta_state[0])
    high_level_obs.append(predict_delta_state[1])
    for vel in predict_delta_state[4:]:
        high_level_obs.append(vel)
    return next_state, np.array(high_level_obs)


def calc_next_input(predict_delta_state, position_buffer):
    predict_size = np.shape(predict_delta_state)
    HL_obs_buffer = np.empty((predict_size[0], predict_size[1]-2))
    for i in range(predict_size[0]):
        position_buffer[i] = position_buffer[i] + predict_delta_state[i][2:4]
        HL_obs_buffer[i][0:2] = predict_delta_state[i][0:2]
        HL_obs_buffer[i][2:] = predict_delta_state[i][4:]
    return HL_obs_buffer, position_buffer


def calc_cost(cost, predict_delta_state, world_buffer, target,latent_action_sample_all, last_velocity):
    for i in range(np.shape(cost)[0]):
        cost[i] = 2 * np.linalg.norm(target[0:2] - world_buffer[i][1:3]) + 1*np.linalg.norm(target[2]- world_buffer[i][0]) # for target reaching
        # cost[i] += 2*np.linalg.norm(target[0:2] - world_buffer[i][3:5]) + 1*np.linalg.norm(target[2]- world_buffer[i][0]) # for velocity tracking
    return cost

def run_mpc_calc_cost(HL_obs_buffer, model, target, latent_action_sample, position_buffer, cost, mean_var):
    latent_action_sample_all = np.swapaxes(latent_action_sample,0,1)
    last_velocity = HL_obs_buffer[0][2:4]
    for i in range(np.shape(latent_action_sample)[0]):
        
        latent_action_norm = normalization(latent_action_sample[i],mean_var[2], mean_var[3] )
        HL_obs_buffer_norm = normalization(HL_obs_buffer, mean_var[0], mean_var[1])
        predict_delta_state_norm = model.predict_para(HL_obs_buffer_norm, latent_action_norm)
        predict_delta_state = inverse_normalization(predict_delta_state_norm,  mean_var[4], mean_var[5])
        HL_obs_buffer, position_buffer = calc_next_input(predict_delta_state, position_buffer)
        cost = calc_cost(cost, predict_delta_state, position_buffer, target,latent_action_sample_all, last_velocity)
    return cost


def get_init_r_yaw(init_foot_pos):
    r_yaw = np.zeros((6,2))
    for i in range(6):
        r_yaw[i][0] = np.linalg.norm(init_foot_pos[i][0:2])
        r_yaw[i][1] = math.atan2(init_foot_pos[i][1], init_foot_pos[i][0])
    return r_yaw


def filter_data(data, win_len = 100):
    '''
        Data: [1,n]
    '''
    filtered_data = []
    print(np.shape(data)[0])
    for i in range(np.shape(data)[1]):
        index = max(0,i-100)
        filtered_data.append(np.mean(data[index:i+1]))
    return np.array(filtered_data)


def easy_cost(target,pre_com_state, post_com_state):
    cost = 2*np.linalg.norm(target[0:2] - post_com_state['base_velocity'][0:2]) + 0*np.linalg.norm(pre_com_state['base_velocity'][0:2] - post_com_state['base_velocity'][0:2])
    # cost = (1 * np.linalg.norm(target[0:2] - np.array(post_com_state['base_pos_x'][0], post_com_state['base_pos_y'][0])) + 
    #             0.2*np.linalg.norm(target[2:]- np.array([math.sin(post_com_state['base_ori_euler'][2]), math.cos(post_com_state['base_ori_euler'][2])] )))
    return cost




