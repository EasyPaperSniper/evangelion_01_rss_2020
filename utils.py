from collections import deque
import random
import os
import math

import torch
import numpy as np
import torch.nn as nn
import gym
import daisy_kinematics

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim,delta_obs_dim, device, capacity):
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

def HL_obs(state):
    '''
    Extract useful information from state(dict) and form to np.array()
    input:
        state: dict
    output:
        HL_obs: np.array
                now the obs includes: com velocity in xyz( need ot translate to the com frame)
                                     joint velocity
    '''
    # TODO: form the HL_obs & translate com velocity to com frame
    high_level_obs = []
    high_level_obs.append(math.sin(state['base_ori_euler'][2])) # yaw information 
    high_level_obs.append(math.cos(state['base_ori_euler'][2]))

    for vel in state['base_velocity'][0:2]:
        high_level_obs.append(vel)

    for j_pos in state['j_pos']:
        high_level_obs.append(j_pos)
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