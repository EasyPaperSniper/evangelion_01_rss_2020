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
        actions = torch.as_tensor(normalization(self.actions[idxs], self.all_mean_var[2], self.all_mean_var[3]), device=self.device).float()
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
        np.save(save_dir+'/idx',np.array([self.idx]))

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
        self.idx = np.load(save_dir+'/idx.npy')[0]

        return self.load_mean_var(save_dir)
        

def HL_obs(state):
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
    high_level_obs.append(math.sin(state['base_ori_euler'][2])) # yaw information 
    high_level_obs.append(math.cos(state['base_ori_euler'][2]))

    for vel in state['base_velocity'][0:2]:
        high_level_obs.append(vel)

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
    high_level_delta_obs.append(math.sin(post_com_state['base_ori_euler'][2]) ) # predict direction 
    high_level_delta_obs.append(math.cos(post_com_state['base_ori_euler'][2]) )

    delta_x = (post_com_state['base_pos_x'][0] - pre_com_state['base_pos_x'][0])
    delta_y = (post_com_state['base_pos_y'][0] - pre_com_state['base_pos_y'][0])
    high_level_delta_obs.append(delta_x) # position changes
    high_level_delta_obs.append(delta_y)
    

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

def calc_next_state(state,predict_delta_state):
    next_state = np.zeros(np.shape(predict_delta_state)[0]-1)
    next_state[0] = math.atan2(predict_delta_state[1], predict_delta_state[0])
    next_state[1] = predict_delta_state[2] + state['base_pos_x'][0]
    next_state[2] = predict_delta_state[3] + state['base_pos_y'][0]
    next_state[3:] =  np.copy(predict_delta_state[4:])

    high_level_obs= []
    high_level_obs.append(predict_delta_state[0])
    high_level_obs.append(predict_delta_state[1])
    for vel in predict_delta_state[4:]:
        high_level_obs.append(vel)
    return next_state, np.array(high_level_obs)


#TODO: fix bug here
def run_mpc_without_norm(state, model, target, latent_action_sample, mean_var):
    high_level_obs = HL_obs(state)
    cost = 0
    for latent_action in latent_action_sample:        
        high_level_obs_norm = normalization(high_level_obs, mean_var[0], mean_var[1])
        latent_action_norm = normalization(latent_action,mean_var[2], mean_var[3] )
        predict_delta_state_norm = model.predict(high_level_obs_norm, latent_action_norm)
        predict_delta_state = inverse_normalization(predict_delta_state_norm, mean_var[4], mean_var[5])
        state, high_level_obs = calc_next_state(state, predict_delta_state)
        # cost += np.linalg.norm(target[0:2] - next_state[3:]) + np.linalg.norm(target[2] - next_state[0])

    # add target position 
    # theta =  0 
    # cost = 4 * np.linalg.norm(target[0:2] - next_state[1:3]) + np.linalg.norm(theta - next_state[0])

    # square circle cost
    cost  =  np.linalg.norm(target[0:2] - state[1:3]) +  np.linalg.norm(target[2:] - high_level_obs[0:2])
    return  high_level_obs


def calc_next_input(predict_delta_state, position_buffer):
    predict_size = np.shape(predict_delta_state)
    HL_obs_buffer = np.empty((predict_size[0], predict_size[1]-2))
    for i in range(predict_size[0]):
        position_buffer[i] = position_buffer[i] + predict_delta_state[i][2:4]
        HL_obs_buffer[i][0:2] = predict_delta_state[i][0:2]
        HL_obs_buffer[i][2:] = predict_delta_state[i][4:]
    return HL_obs_buffer, position_buffer


def calc_cost(cost, predict_delta_state, position_buffer, target,latent_action_sample_all=None):
    for i in range(np.shape(cost)[0]):
        cost[i] += (10 * np.linalg.norm(target[0:2] - predict_delta_state[i][4:6]) + 0.0*np.linalg.norm( predict_delta_state[i][0]) + 
                    0.0*np.std(latent_action_sample_all[i])) - 1*np.linalg.norm(predict_delta_state[i][5:6])
        # cost[i] =   (4 * np.linalg.norm(target[0:2] - position_buffer[i]) + 1*np.linalg.norm(target[0:2]/10 - predict_delta_state[i][4:6]) 
        #             + 0*np.linalg.norm(math.atan2(predict_delta_state[i][1],predict_delta_state[i][0])))
        # cost[i] =   (3 * np.linalg.norm(target[0:2] - position_buffer[i]) + 0*np.linalg.norm(predict_delta_state[i][4:6]) 
        #             + 1*np.linalg.norm(math.atan2(predict_delta_state[i][1],predict_delta_state[i][0])))
        # cost[i] =   (3 * np.linalg.norm(target[0:2] - position_buffer[i]) + 0*np.linalg.norm(predict_delta_state[i][4:6]) 
        #             + 0*np.linalg.norm(math.atan2((target[0:2] - position_buffer[i])[1],(target[0:2] - position_buffer[i])[0])-math.atan2(predict_delta_state[i][1],predict_delta_state[i][0])))
        # cost[i] = 3 * np.linalg.norm(target[0:2] - position_buffer[i]) + 5*np.linalg.norm(target[2:]- predict_delta_state[i][0:2]) 
    return cost

def run_mpc_calc_cost(HL_obs_buffer, model, target, latent_action_sample, position_buffer, cost, mean_var):
    latent_action_sample_all = np.swapaxes(latent_action_sample,0,1)
    for i in range(np.shape(latent_action_sample)[0]):
        latent_action_norm = normalization(latent_action_sample[i],mean_var[2], mean_var[3] )
        HL_obs_buffer_norm = normalization(HL_obs_buffer, mean_var[0], mean_var[1])
        predict_delta_state_norm = model.predict_para(HL_obs_buffer_norm, latent_action_norm)
        predict_delta_state = inverse_normalization(predict_delta_state_norm,  mean_var[4], mean_var[5])
        HL_obs_buffer, position_buffer = calc_next_input(predict_delta_state, position_buffer)
        cost = calc_cost(cost, predict_delta_state, position_buffer, target,latent_action_sample_all)
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





