import numpy as np
import torch


class forward_model():
    '''
    The forward model(mid-level policy) is a NN which is trained in a supervised manner
    '''
    def __init__(self,):
        '''
        Initialize the structure and options for model
        '''

    def supervised_learning(self):
        return 
    
    def predict(self,state, action):
        return state

    
    def curiosity_reward(self, next_state):
        return False



def sinusoidal_action(self, params,t):
    '''
    Args:
        params(array): parameters for controller 54*1
        t(float): phase variable
    Returns:
        action(array): action acting on robot 18*1
    '''

    return action 


class rs_learning():
    def __init__(self,la_dim):
        '''
        Initialize policy parameters and their distribution
        Initialize learning parameters
        Initialize other stuff
        '''
        #TODO: need to initialize dataset for training forward_model

    self.policy_parameter = np.zeros(la_dim)
    self.forward_model = forward_model()
    self.lr = lr
    
    def update_policy(self):


    def update_model(self):

    
    def update_all(self):

    
    def save_model(self):

    def save_policy(self):
    
    def save_all(self):
    
    def load_model(self):

    def load_policy(self):

    def load_all(self):
    
    