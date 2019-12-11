from collections import deque
import random
import os

import torch
import numpy as np
import torch.nn as nn
import gym

class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, device, capacity):
        self.device = device
        self.capacity = capacity

        if type(obs_dim) == int:
            self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
            self.next_obses = np.empty((capacity, obs_dim), dtype=np.float32)
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


def high_level_obs(state):
    '''
    Extract useful information from state(dict) and form to np.array()
    input:
        state: dict
    output:
        HL_obs: np.array
    '''
    # TODO: form the HL_obs
    HL_obs = state['j_pos']
    return HL_obs

def high_level_delta_obs(state): #these two functions can be combined to be one func
    '''
    Extract useful information from state(dict) and form to np.array()
    input:
        state: dict
    output:
        HL_obs: np.array
    '''
    # TODO: form the HL_obs
    HL_delta_obs = state['j_pos']
    return HL_delta_obs


def check_robot_dead(state):
    '''
    check if the robot falls
    input:
        state: dict
    output:
        bool
    '''
    #TODO: define termination condition
    return 