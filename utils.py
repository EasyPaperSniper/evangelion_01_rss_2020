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
    def __init__(self, obs_dim, action_dim,delta_obs_dim, device, capacity, save_dir = None):
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

        obses = torch.as_tensor(normalization(self.obses[idxs], self.all_mean_var[0], self.all_mean_var[1]), device=self.device).float()
        actions = torch.as_tensor(normalization(self.actions[idxs], self.all_mean_var[2], self.all_mean_var[3]), device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            normalization(self.next_obses[idxs], self.all_mean_var[4], self.all_mean_var[5]), device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save_buffer(self, save_dir):
        save_dir = make_dir(os.path.join(save_dir + '/buffer_data'))
        np.save(save_dir+'/obses', self.obses)
        np.save(save_dir+'/actions', self.actions)
        np.save(save_dir+'/rewards', self.rewards)
        np.save(save_dir+'/next_obses', self.next_obses)
        np.save(save_dir+'/not_dones', self.not_dones)

        # save mean and var
        self.all_mean_var[0]= np.mean(self.obses, axis = 0)
        self.all_mean_var[1]= np.std(self.obses, axis = 0)
        self.all_mean_var[2]= np.mean(self.actions, axis = 0)
        self.all_mean_var[3]= np.std(self.actions, axis = 0)
        self.all_mean_var[4]= np.mean(self.next_obses, axis = 0)
        self.all_mean_var[5]= np.std(self.next_obses, axis = 0)
        np.save(save_dir+'/all_mean_var', self.all_mean_var)

    def load_mean_var(self,save_dir):
        '''
        0: mean of obses; 1: var of obses; 2: mean of actions; 3: var of actions; 4: mean of next_obses; 5: var of next_obses
        '''
        self.all_mean_var = np.load(save_dir+'/all_mean_var.npy')
        return self.all_mean_var

    def load_buffer(self, save_dir):
        save_dir = save_dir +'/buffer_data'
        self.obses = np.load(save_dir+'/obses.npy')
        self.actions = np.load(save_dir+'/actions.npy')
        self.rewards = np.load(save_dir+'/rewards.npy')
        self.next_obses = np.load(save_dir+'/next_obses.npy')
        self.not_dones = np.load(save_dir+'/not_dones.npy')
        self.idx = np.shape(self.obses)[0]
        return self.load_mean_var(save_dir)
        

def HL_obs(state):
    '''
    Extract useful information from state(dict) and form to np.array()
    input:
        state: dict
    output:
        HL_obs: np.array
                now the obs includes: com velocity in xyz, yaw information, footplace location in the CoM frame
    '''
    # TODO: form the HL_obs & translate com velocity to com frame
    high_level_obs = []
    high_level_obs.append(math.sin(state['base_ori_euler'][2])) # yaw information 
    high_level_obs.append(math.cos(state['base_ori_euler'][2]))

    for vel in state['base_velocity'][0:2]:
        high_level_obs.append(vel)

    foot_pose_world = daisy_raibert_controller.get_foot_position_world_from_com(state)
    for foot in foot_pose_world:
        for foot_pos in foot[0:2]:
            high_level_obs.append(foot_pos)

    return np.array(high_level_obs)

def HL_delta_obs(pre_com_state,post_com_state): 
    '''
    Extract useful information from state(dict) and form to np.array()
    input:
        state: dict
    output:
        HL_obs: np.array
    '''
    # TODO: form the HL_obs
    high_level_delta_obs = []
    high_level_delta_obs.append(post_com_state['base_ori_euler'][2] ) # predict direction 

    high_level_delta_obs.append(post_com_state['base_pos_x'][0] - pre_com_state['base_pos_x'][0]) # position changes
    high_level_delta_obs.append(post_com_state['base_pos_y'][0] - pre_com_state['base_pos_y'][0])
   
    for vel in post_com_state['base_velocity'][0:2]:# velocity information
        high_level_delta_obs.append(vel)
    
    foot_pose_world = daisy_raibert_controller.get_foot_position_world_from_com(post_com_state)
    for foot in foot_pose_world:
        for foot_pos in foot[0:2]:
            high_level_delta_obs.append(foot_pos)

    return np.array(high_level_delta_obs)


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

def check_data_useful(state):
    return True

def calc_next_state(state,post_state):
    next_state = np.array(post_state)
    next_state[1] = post_state[1] + state['base_pos_x'][0]
    next_state[2] = post_state[2] + state['base_pos_y'][0]

    pre_com_state= []
    pre_com_state.append(math.sin(post_state[0]))
    pre_com_state.append(math.cos(post_state[0]))
    for info in post_state[3:]:
        pre_com_state.append(info)
    return next_state, pre_com_state


#TODO: fix bug here
def run_mpc(state, model, cost_func, latent_action_sample):
    pre_com_state = HL_obs(state)
    for latent_action in latent_action_sample:
        post_state = model.forward(pre_com_state, latent_action)
        next_state, pre_com_state = calc_next_state(state, post_state)
    cost = cost_func(next_state)
    return cost