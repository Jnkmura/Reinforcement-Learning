import numpy as np
import tensorflow as tf 
import threading
import gym
import os
from time import time
from keras.layers import Input, Dense, Activation, LSTM, Reshape
from keras.models import Model
from gym.spaces import Box, Discrete

def get_cumulative_rewards(rewards, value, gamma):              
    value_target = []
    for r in rewards[::-1]:    
        value = r + (gamma * value)
        value_target.append(value)
    cumulative = list(reversed(value_target))
    return cumulative

def get_act_dim(env):
    action_space = env.action_space
    if isinstance(action_space, Discrete):
        action_n = action_space.n
    else:
        action_n = action_space.shape[0]
    return action_space, action_n

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


class Env:
    def __init__(self, env_name, env_n, seed=1):
        self.env_n = env_n
        self.env_name = env_name
        self.seed = seed
        self.envs = self.create_envs()

    def create_envs(self):
        envs = []
        for n in range(self.env_n):
            env = gym.make(self.env_name)
            env.seed(self.seed + n)
            envs.append(env)
        return envs

class Network:
    def __init__(self, env, isglobal = False, globalnet = None, gamma=0.90, critic_lr = 1e-3, actor_lr = 1e-4):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        _, action_n = get_act_dim(env)
        if isinstance(env.action_space, Box):
            upper_bound = env.action_space.high
            lower_bound = env.action_space.low
        
        self.state_ph, self.action_ph = space_action_placeholders(env)
        self.target_ph = tf.placeholder(tf.float32, shape = (None, 1))

        if isglobal:
            _, self.critic_var, _ = self.define_critic(self.state_ph)
            _, self.actor_var, _ = self.define_actor(self.state_ph, action_n)

        else:
            self.critic_model, self.critic_var, self.critic_opt = self.define_critic(self.state_ph)
            self.value = self.critic_model.output
            self.td = self.target_ph - self.value
            self.critic_loss = tf.reduce_mean(self.td ** 2)

            self.actor_model, self.actor_var, self.actor_opt = self.define_actor(self.state_ph, action_n)
            mu, sigma = self.actor_model.outputs
            mu, sigma = mu * upper_bound, sigma + 1e-4
            
            dist = tf.distributions.Normal(mu, sigma)
            logprob = dist.log_prob(self.action_ph)
            entropy = dist.entropy()
            self.exp_v = (logprob * tf.stop_gradient(self.td)) + (entropy * 0.01)
            self.actor_loss = tf.reduce_mean(-self.exp_v)
            
            self.action = tf.clip_by_value(
                tf.squeeze(dist.sample(1), axis = [0, 1]), lower_bound, upper_bound)

            self.actor_grads = tf.gradients(
                self.actor_loss, self.actor_var)
            self.critic_grads = tf.gradients(
                self.critic_loss, self.critic_var)

            self.update_actor = self.actor_opt.apply_gradients(
                zip(self.actor_grads, globalnet.actor_var))
            self.update_critic = self.critic_opt.apply_gradients(
                zip(self.critic_grads, globalnet.critic_var))

            self.actor_from_global = [av.assign(ga) for av, ga in zip(
                self.actor_var, globalnet.actor_var)]
            self.critic_from_global = [cv.assign(gc) for cv, gc in zip(
                self.critic_var, globalnet.critic_var)]

    def define_actor(self, state_ph, actions_n):
        inputs = Input(tensor = state_ph)
        x = Dense(200, activation = 'relu')(inputs)
        x = Dense(200, activation = 'relu')(x)
        #x = Reshape((1, 200))(x)
        #x = Dense(100, activation = 'relu')(x)
        #x = LSTM(100, return_sequences = True)(x)
        #x = LSTM(100)(x)
        mu = Dense(actions_n, activation = 'tanh')(x)
        sigma = Dense(actions_n, activation = 'softplus')(x)

        actor_model = Model(inputs=[inputs], outputs=[mu, sigma])
        actor_var = actor_model.weights
        actor_opt = tf.train.AdamOptimizer(self.actor_lr)
        return actor_model, actor_var, actor_opt

    def define_critic(self, state_ph):
        inputs = Input(tensor = state_ph)
        x = Dense(100, activation = 'relu')(inputs)
        x = Dense(100, activation = 'relu')(x)
        #x = Reshape((1, 100))(x)
        #x = Dense(100, activation = 'relu')(x)
        #x = LSTM(100, return_sequences = True)(x)
        #x = LSTM(100)(x)
        value = Dense(1)(x)

        critic_model = Model(inputs=[inputs], outputs = [value])
        critic_var = critic_model.weights
        critic_opt = tf.train.AdamOptimizer(self.critic_lr)
        return critic_model, critic_var, critic_opt

class Agent:
    def __init__(self, env, globalnet = None, agent_name = "Agent"):
        self.env = env
        self.env_name = env.spec.id
        self.agent_name = agent_name
        self.Network = Network(self.env, isglobal=False, globalnet=globalnet)

    def play(self, max_episodes = 1000, max_steps = 1000):
        writer = tf.summary.FileWriter(
        os.path.join('logs', self.env_name, self.agent_name, str(time())))
        episodes = 1
        total_steps = 0
        states, actions, rewards = [], [], []
        while episodes <= max_episodes:
            episodes_rewards = 0
            state = self.env.reset()
            for _ in range(max_steps):
                total_steps += 1
                action = sess.run(
                    self.Network.action, {self.Network.state_ph: state[None]})
                next_state, reward, done, _ = self.env.step(action)
                episodes_rewards += reward

                states.append(state)
                actions.append(action)
                #rewards.append(max(min(float(reward), 1.0), -1.0))
                rewards.append((reward+8)/8)
                update_by_steps = total_steps % 5 == 0

                if done or update_by_steps:
                    value = 0
                    if update_by_steps:
                        value = sess.run(self.Network.value,
                         {self.Network.state_ph: next_state[None]})[0][0]

                    cumulative = get_cumulative_rewards(
                        rewards, value, self.Network.gamma)
                    sess.run(
                        [self.Network.update_actor, self.Network.update_critic],
                        {
                            self.Network.state_ph: np.vstack(states),
                            self.Network.action_ph: np.vstack(actions),
                            self.Network.target_ph: np.vstack(cumulative)
                            } 
                    )

                    state = next_state
                    states, actions, rewards = [], [], []
                    sess.run([self.Network.actor_from_global,
                              self.Network.critic_from_global])

                    if done:
                        episodes += 1
                        summary = tf.Summary()
                        summary.value.add(
                            tag='Episode Rewards', simple_value = episodes_rewards)
                        writer.add_summary(summary, episodes)
                        break

if __name__ == "__main__":
    global GLOBALNET
    coord = tf.train.Coordinator()
    env_name = "Pendulum-v0"
    #env_name = "BipedalWalker-v3"
    global_env = gym.make(env_name)
    GLOBALNET = Network(global_env, isglobal=True)
    agents_num = 3
    envs = Env(env_name, agents_num, seed=np.random.randint(1, 99999))

    agents = []
    for i in range(agents_num):
        agent_name = "Agent" + str(i)
        agent = Agent(envs.envs[i], GLOBALNET, agent_name)
        agents.append(agent)
   
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    threads = []
    for agent in agents:
        job = lambda: agent.play(10000)
        t = threading.Thread(target=job)
        t.start()
        threads.append(t)
    coord.join(threads)

