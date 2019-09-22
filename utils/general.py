import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete

def get_placeholders(*args):
    return [tf.placeholder(tf.float32, shape = (None,)) for arg in args]

def space_action_placeholders(env):
    action_space = env.action_space
    shape = (None,)
    state_ph = tf.placeholder(
        tf.float32, shape = shape + env.observation_space.shape)

    if isinstance(action_space, Discrete):
        action_ph = tf.placeholder(
            tf.int32, shape = shape)
    else:
        action_ph = tf.placeholder(
            tf.float32, shape = shape + action_space.shape)
    return state_ph, action_ph    
            
def get_act_dim(env):
    action_space = env.action_space
    if isinstance(action_space, Discrete):
        action_n = action_space.n
    else:
        action_n = action_space.shape[0]
    return action_space, action_n

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)