import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils

LOG_FREQ = 10000
OUT_SIZE = 29


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def gaussian_likelihood(noise, log_std):
    pre_sum = -0.5 * noise.pow(2) - log_std
    return pre_sum.sum( -1, keepdim=True) - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def apply_squashing_func(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max, ):
        super().__init__()
        self.encoder = None

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),)
            # nn.Linear(hidden_dim, 2 * action_dim))
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self,
                obs,
                compute_pi=True,
                compute_log_pi=True,
                detach_encoder=False):

        # mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        x = self.trunk(obs)
        mu = self.mean_linear(x)
        log_std = self.std_linear(x)
        
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
            log_det = log_std.sum(dim=-1)
            entropy = 0.5 * (1.0 + math.log(2 * math.pi) + log_det)
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_likelihood(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = apply_squashing_func(mu, pi, log_pi)

        return mu, pi, log_pi, entropy



class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)



class Critic(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim,
                 log_std_min, log_std_max,):
        super().__init__()


        self.Q1 = QFunction(obs_dim, action_dim, hidden_dim)
        self.Q2 = QFunction(obs_dim, action_dim, hidden_dim)

        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        return q1, q2, obs


class SAC(object):
    def __init__(self, device, obs_dim, action_dim, hidden_dim,
                 initial_temperature=0.1, actor_lr=1e-3, critic_lr=1e-3, actor_beta=0.9,
                 critic_beta=0.9, log_std_min=-20, log_std_max=2,):
        self.device = device

        self.actor = Actor(
            obs_dim,
            action_dim,
            hidden_dim,
            log_std_min,
            log_std_max,).to(device)

        self.critic = Critic(
            obs_dim,
            action_dim,
            hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,).to(device)

        self.critic_target = Critic(
            obs_dim,
            action_dim,
            hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))

        self.log_alpha = torch.tensor(np.log(initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def plan_latent_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False)

            return mu.cpu().data.numpy().flatten()

    def sample_latent_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)

            return pi.cpu().data.numpy().flatten()


    def _update_critic(self, obs, action, reward, next_obs, not_done, discount,):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2, _ = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * discount * target_V)

        # Get current Q estimates
        current_Q1, current_Q2, h_obs = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def _update_actor(self, obs, target_entropy,):
        mu, pi, log_pi, entropy = self.actor(obs, detach_encoder=True)


        actor_Q1, actor_Q2, _ = self.critic(obs, pi, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean() 

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        if target_entropy is not None:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()


    def update_policy(self,
               replay_buffer,
               batch_size=32,
               discount=0.99,
               tau=0.005,
               target_entropy=-10):

        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)
        self._update_critic(obs, action, reward, next_obs, not_done, discount,)
        self._update_actor(obs, target_entropy,)
        soft_update_params(self.critic, self.critic_target, tau)


    def save(self, model_dir):
        torch.save(self.actor.state_dict(),
                   '%s/actor.pt' % (model_dir))
        torch.save(self.critic.state_dict(),
                   '%s/critic.pt' % (model_dir))

    def load(self, model_dir):
        self.actor.load_state_dict(
            torch.load('%s/actor.pt' % (model_dir)))
        self.critic.load_state_dict(
            torch.load('%s/critic.pt' % (model_dir)))
