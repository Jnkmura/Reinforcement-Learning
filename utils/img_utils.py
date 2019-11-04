
from gym.core import Wrapper, ObservationWrapper
from scipy.misc import imresize
from gym.spaces import Box, Discrete
import numpy as np

class FrameBuffer(Wrapper):
    # stack 4 frames into one state
    def __init__(self, env, n_frames=4):
        super(FrameBuffer, self).__init__(env)
        height, width, n_channels = env.observation_space.shape
        obs_shape = [height, width, n_channels * n_frames]
        self.observation_space = Box(0.0, 1.0, obs_shape, dtype=np.float32)
        self.framebuffer = np.zeros(obs_shape, 'float32')
        
    def reset(self):
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer
    
    def step(self, action):
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info
    
    def update_buffer(self, img):
        offset = self.env.observation_space.shape[-1]
        cropped_framebuffer = self.framebuffer[:,:,:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis = -1)


class Preprocess(ObservationWrapper):
    def __init__(self, env, img_size = (84, 84, 1)):
        ObservationWrapper.__init__(self, env)
        self.img_size = img_size
        if not self.img_size:
            self.img_size = (env.observation_space.shape[0],
                             env.observation_space.shape[1],
                             1)
                             
        self.observation_space = Box(0.0, 1.0, self.img_size
                                     , dtype=np.float32)

    def observation(self, img):        
        
        # resize and normalize img
        img = img[35:195, 0:160]
        img = imresize(img, self.img_size)
        img = img.mean(-1, keepdims=True)
        img = img.astype('float32') / 255.
        
        return img