import gym
import numpy as np
import random
import tensorflow as tf
import keras
from time import time
import os
import argparse
import logging
import sys
from tqdm import tqdm
from keras.models import Model

sys.path.insert(0, '../')
from utils.models import generic_model
from utils.experience_replay import ExpReplay
from utils.img_utils import FrameBuffer, Preprocess
from utils.general import get_placeholders, space_action_placeholders, get_act_dim

class DQN:
    def __init__(self, env_name, epsilon=0, gamma = 0.99, lr = 1e-4,  double = ""):

        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.env_name = env_name
        env = self.create_env()
        _, action_n = get_act_dim(env)
       
        # placeholders that will be used to do the training
        self.rewards_ph, self.is_done_ph = get_placeholders(None, None)
        self.obs_ph, self.action_ph = space_action_placeholders(env)
        self.next_obs_ph = tf.placeholder('float32', shape=(None,) + env.observation_space.shape)
        self.is_not_done = 1 - self.is_done_ph

        inputs, self.output = generic_model(self.obs_ph, action_n, 'linear', 'relu')
        self.network = tf.keras.models.Model(inputs, self.output)

        #target network
        self.target_network = tf.keras.models.clone_model(self.network) 

        # q-values based on choosed actions
        self.action_qvalues = tf.reduce_sum(
            tf.one_hot(self.action_ph, action_n) * self.network(self.obs_ph), axis=1)
        if double:
            # Use DDQN here
            self.next_qvalues = self.target_network(self.next_obs_ph)
            self.next_qvalues_agent = self.network(self.next_obs_ph)
            self.next_action = tf.argmax(self.next_qvalues_agent, axis = 1)
            self.next_max_qsa = tf.reduce_sum(
                self.next_qvalues * tf.one_hot(self.next_action, action_n), axis = 1)
        else:
            self.next_max_qsa = tf.reduce_max(
                self.target_network(self.next_obs_ph), axis=1)
            
        # Target q-value
        self.target_values = self.rewards_ph + self.gamma * tf.multiply(
            self.next_max_qsa, self.is_not_done)
        # Loss func
        self.td_loss = tf.reduce_mean(
            tf.losses.huber_loss(
                labels=self.target_values,
                predictions=self.action_qvalues))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(
            self.td_loss, var_list=self.network.weights)
        sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter(os.path.join('logs', self.env_name, str(time())))

    def create_env(self):
        env = gym.make(self.env_name)
        if len(env.observation_space.shape) > 2:
            env = Preprocess(env)
            env = FrameBuffer(env, n_frames=4)
        env.reset()
        return env

    def get_action(self, env, state):
        q_values = sess.run(self.output, {self.obs_ph: state[None]})
        action = env.action_space.sample()
        if np.random.random() <= self.epsilon:
            return action
        return np.argmax(q_values)
    
    def transfer_weights(self):
        self.target_network.set_weights(self.network.get_weights())

    def evaluate(self, render=False, monitor=False, greedy = False,
                       episodes = 20):
        env = self.create_env()
        rewards = []
        
        for e in range(episodes):
            if monitor:
                env = gym.wrappers.Monitor(
                    env,
                    os.path.join('video', str(time())),
                    video_callable=lambda episode_id: True,
                    force=True)
            state = env.reset()
            reward = 0
            while True:
                qvalues = sess.run(self.output, {self.obs_ph: state[None]})
                a = qvalues.argmax(axis=-1)[0] if greedy else self.get_action(env, state)
                next_s, r, done, _ = env.step(a)
                if render:
                    env.render()
                reward += r
                state = next_s
                if done:
                    print('Evaluation Reward: {}'.format(reward))
                    rewards.append(reward)
                    break
                    
            env.close()
        avg_rewards =  np.mean(rewards)
        print('mean rewards over {} episodes: {}'.format(episodes, avg_rewards))
        return avg_rewards

    def play(self, env, exp_replay, n_steps=25, custom_reward = 0, stop = False):
        if len(env.observation_space.shape) > 2:
            state = env.framebuffer
        else:
            state = env.env.state

        if state is None:
            state = env.reset()
        else:
            state = np.array(state)
        
        last_info = None
        total_reward = 0
        for steps in range(n_steps):
            a = self.get_action(env, state)
            next_s, r, done, info = env.step(a)
            if custom_reward != 0:
                if (last_info is not None and last_info['ale.lives'] > info['ale.lives']):
                    r = custom_reward
            exp_replay.add(state, a, r, next_s, done)
            total_reward += r
            
            if done:
                state = env.reset()
                last_info = None
                if stop:
                    break
            else:
                state = next_s
                last_info = info
                
        return total_reward

    def train(self,
            exp_replay,
            iterations,
            steps = 25,
            batch = 64,
            eps_decrease = 0.9,
            games_eval = 3,
            eval_freq = 100,
            transf_freq = 500,
            eps_decrease_freq = 500
            ):

        env = self.create_env()
        env.reset()
        eval = 0
   
        for i in tqdm(range(iterations)):
            self.play(env, exp_replay, n_steps = steps, custom_reward = -10)

            obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch)
            sess.run([self.train_step, self.td_loss], {
                                    self.obs_ph:obs_batch
                                    ,self.action_ph:act_batch
                                    ,self.rewards_ph:reward_batch
                                    ,self.next_obs_ph:next_obs_batch
                                    ,self.is_done_ph:is_done_batch
                                })

            if i % transf_freq == 0:
                self.transfer_weights()

            if i % eps_decrease_freq == 0:
                self.epsilon = max(self.epsilon * eps_decrease, 0.01) 

            if i % eval_freq == 0:
                eval += 1
                mean_rw = self.evaluate(episodes = games_eval)
                summary=tf.Summary()
                summary.value.add(tag='Mean Evaluations', simple_value = mean_rw)
                self.writer.add_summary(summary, eval)
                print(self.epsilon)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--enviroment', default='PongDeterministic-v4')
    parser.add_argument('-exp_size', '--exp_size', default=50000)
    parser.add_argument('-epsilon', '--epsilon', default=0.5)
    parser.add_argument('-d', '--double', default="")
    parser.add_argument('-i', '--iterations', default=100000)
    parser.add_argument('-eps_d', '--eps_decrease', default=0.995)
    parser.add_argument('-warm', '--warm', default=1)
    parser.add_argument('-transf_freq', '--transf_freq', default=500)
    parser.add_argument('-eps_decrease_freq', '--eps_decrease_freq', default=500)
    parser.add_argument('-steps_per_iteration', '--steps_per_iteration', default=25)
    args = vars(parser.parse_args())

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)

    dqn = DQN(
        args['enviroment'],
        epsilon=float(args['epsilon']),
        double=int(args['exp_size']))

    size = int(args['exp_size'])
    warm = int(args['warm'])
    exp_replay = ExpReplay(size)
    if warm:
        print('Warming up...')
        dqn.play(dqn.create_env(), exp_replay, n_steps=size, stop = False)
        print('Expsize: {}'.format(len(exp_replay)))

    print(args['eps_decrease'])
    dqn.train(
        exp_replay,
        int(args['iterations']),
        eps_decrease = float(args['eps_decrease']),
        transf_freq = int(args['transf_freq']),
        eps_decrease_freq = int(args['eps_decrease_freq']),
        steps = int(args['steps_per_iteration'])
        )

    dqn.evaluate(monitor=True, greedy=True)

