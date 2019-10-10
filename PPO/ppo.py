import tensorflow as tf
import gym
import time
import numpy as np
import os, sys
import argparse

sys.path.insert(0, '../')
from keras.layers import Input, Dense, Activation, Conv2D, Flatten
from keras.models import Model
from keras.layers.merge import concatenate, Add
from gym.spaces import Box, Discrete
from utils.experience_replay import PPOBuffer
from utils.img_utils import FrameBuffer, Preprocess
from utils.general import get_placeholders, space_action_placeholders, get_act_dim

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

class Policy:
    def __init__(self, state_ph, action_ph, action_space, action_n):
        self.action_space = action_space
        if len(state_ph.shape.as_list()) > 2:
            x = self.conv_layers(state_ph, action_n)
            self.value = self.conv_layers(state_ph, 1)
        else:
            x = self.define_layers(state_ph, action_n)
            self.value = self.define_layers(state_ph, 1)
            
        self.pi, self.logp, self.logp_pi = self.define_policy(x, action_ph, action_space)

    def conv_layers(self, state_ph, action_n, ouput_activation = None):
        inputs = Input(tensor = state_ph)
        x = Conv2D(filters=32, kernel_size=8, strides=4
                                ,activation='tanh'
                                ,padding='valid'
                                ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(inputs) 
        x = (Conv2D(filters=64, kernel_size=4, strides=2
                                ,activation='tanh'
                                ,padding='valid'
                                ,kernel_initializer=tf.variance_scaling_initializer(scale=2)))(x) 
        x = Conv2D(filters=64, kernel_size=3, strides=1
                                ,activation='tanh'
                                ,padding='valid'
                                ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu'
        ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(x) 
        x = Dense(action_n, activation = ouput_activation
        ,kernel_initializer=tf.variance_scaling_initializer(scale=2))(x)
        return x 
        
    def define_layers(self, state_ph, action_n, ouput_activation = None):
        inputs = Input(tensor = state_ph)
        x = Dense(100, activation = 'tanh')(inputs)
        x = Dense(100, activation = 'tanh')(x)
        x = Dense(100, activation = 'tanh')(x)
        x = Dense(action_n, activation = ouput_activation)(x)
        return x
        
    def define_policy(self, x, action_ph, action_space):
        if isinstance(action_space, Box):
            act_dim = action_ph.shape.as_list()[-1]
            log_std = tf.get_variable(
                name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
            std = tf.exp(log_std)
            pi = x + tf.random.normal(tf.shape(x)) * std
            logp = gaussian_likelihood(action_ph, x, log_std)
            logp_pi = gaussian_likelihood(pi, x, log_std)
            return pi, logp, logp_pi
        
        if isinstance(action_space, Discrete):
            act_dim = action_space.n
            logp_all = tf.nn.log_softmax(x)
            pi = tf.squeeze(tf.multinomial(x, 1), axis=1)
            logp = tf.reduce_sum(tf.one_hot(action_ph, depth=act_dim) * logp_all, axis=1)
            logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
            return pi, logp, logp_pi
        
    def return_main_values(self):
        return self.pi, self.logp, self.logp_pi, self.value 

class PPO:
    def __init__(self, 
                 env_name,
                 steps_per_epoch=4000,
                 epochs=50,
                 gamma=0.99,
                 clip_ratio=0.2,
                 pi_lr=3e-4,
                 vf_lr=1e-3,
                 train_pi_iters=80,
                 train_v_iters=80,
                 lam=0.97,
                 max_ep_len=10000,
                 target_kl=0.01):
        
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.env_name = env_name

        env = self.create_env()
        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape

        action_space, action_n = get_act_dim(env)
        self.state_ph, self.action_ph = space_action_placeholders(env)
        self.adv_ph, self.ret_ph, self.logp_old_ph = get_placeholders(None, None, None)
        
        policy = Policy(self.state_ph, self.action_ph, action_space, action_n)
        self.pi, self.logp, self.logp_pi, self.v = policy.return_main_values()
        
        # Experience buffer
        self.buf = PPOBuffer(
            state_dim, action_dim, self.steps_per_epoch, self.gamma, self.lam)

        # Losses
        ratio = tf.exp(self.logp - self.logp_old_ph)    
        min_adv = tf.where(self.adv_ph > 0, (1 + self.clip_ratio) * self.adv_ph, (1 - self.clip_ratio) * self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)      

        # Optimizers
        self.train_pi = tf.train.AdamOptimizer(learning_rate = self.pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(learning_rate = self.vf_lr).minimize(self.v_loss)

    def play(self):
        env = self.create_env()
        ep = 0
        start_time = time.time()
        writer = tf.summary.FileWriter(os.path.join('logs', self.env_name, str(start_time)))
        state, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        for epoch in range(self.epochs):
            for t in range(self.steps_per_epoch):
                a, v_t, logp_t = sess.run([self.pi, self.v, self.logp_pi], feed_dict={self.state_ph: state[None]})

                self.buf.store(state, a, reward, v_t, logp_t)
                state, reward, done, _ = env.step(a[0])
                ep_ret += reward
                ep_len += 1

                terminal = done or (ep_len == self.max_ep_len)
                if terminal or (t==self.steps_per_epoch-1):
                    last_val = reward if done else sess.run(self.v, feed_dict={self.state_ph: state[None]})
                    self.buf.finish_path(last_val)
                    env.close()
                    
                    ep += 1
                    summary=tf.Summary()
                    summary.value.add(tag='Episode Rewards', simple_value = ep_ret)
                    writer.add_summary(summary, ep)

                    print('Episode {} - Reward: {}'.format(ep, ep_ret))
                    
                    #summary=tf.Summary()
                    #summary.value.add(tag='Episode Evalution', simple_value = self.evaluate())
                    #writer.add_summary(summary, ep)
                     
                    state, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            self.train()
        
    def train(self):
        state_buf, act_buf, adv_buf, ret_buf, logp_buf = self.buf.get()
        inputs = {self.state_ph: state_buf,
                  self.action_ph: act_buf,
                  self.adv_ph: adv_buf,
                  self.ret_ph: ret_buf,
                  self.logp_old_ph: logp_buf}
        
        for i in range(self.train_pi_iters):
            _, kl = sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            kl = np.mean(kl)
            if kl > 1.5 * self.target_kl:
                break
        for _ in range(self.train_v_iters):
            sess.run(self.train_v, feed_dict=inputs)
        
    def evaluate(self, render=False, monitor=False, episodes = 20):
        env = self.create_env()
        rewards = []
        
        for e in range(episodes):
            if monitor:
                env = gym.wrappers.Monitor(
                    env,
                    os.path.join('video', str(time.time())),
                    video_callable=lambda episode_id: True,
                    force=True)
            s = env.reset()
            reward = 0
            while True:
                a = sess.run([self.pi], feed_dict={self.state_ph: s[None]})
                next_s, r, done, _ = env.step(a[0][0])
                if render:
                    env.render()
                reward += r
                s = next_s
                if done:
                    print('Evaluation Reward: {}'.format(reward))
                    rewards.append(reward)
                    break
                    
            env.close()
        avg_rewards =  np.mean(rewards)
        print('mean rewards over {} episodes: {}'.format(episodes, avg_rewards))
        return avg_rewards

    def create_env(self):
        env = gym.make(self.env_name)
        if len(env.observation_space.shape) > 2:
            env = Preprocess(env)
            env = FrameBuffer(env, n_frames=4)
        return env

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--enviroment', default='BipedalWalker-v2')
    parser.add_argument('-e', '--epochs', default=50)
    parser.add_argument('-cpu', '--cpu', default=2)
    args = vars(parser.parse_args())

    epochs = int(args['epochs'])
    ppo = PPO(args['enviroment'], epochs = epochs)

    session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=int(args['cpu']),
      inter_op_parallelism_threads=int(args['cpu']))
    sess = tf.Session(config=session_conf)
    sess.run(tf.global_variables_initializer())
    ppo.play()

    ppo.evaluate(render=True, monitor=True)