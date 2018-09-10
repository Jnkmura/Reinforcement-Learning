import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class Qlearning(object):
    
    def __init__ (self, epsilon, alpha, gamma, legal_actions):
        
        self.epsilon = epsilon  # percentage of random actions
        self.alpha = alpha      # percentage of update training
        self.gamma = gamma      # discount factor for training
        self.legal_actions = legal_actions # every legal action for each state
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0)) # our qtable
        
    def get_next_state_value(self, state):

        # calculating V(s')
        
        actions = self.legal_actions(state)
        A = np.zeros(len(actions))
        if len(actions) == 0:
            return 0
        
        for i,a in enumerate(actions):
            
            A[i] = self._qvalues[state][a]
            
        return max(A)
    
    def train_qsa(self, state, action, reward, next_state):

        # updating our qtable based on our moves
        # (1 - alpha) * Q(s,a) + (alpha * (r + gamma * V(s'))
        
        current_qsa = self._qvalues[state][action]
        next_qsa = self.get_next_state_value(next_state)
        
        new_qsa = ((1 - self.alpha)*current_qsa) + (self.alpha * (reward + (self.gamma * next_qsa)))
        
        self._qvalues[state][action] = new_qsa
        
    
    def get_action(self, state):
        
        # Lets find the best action first
        actions = self.legal_actions(state)
        if len(actions) == 0:
            return None
        
        # array with Q(s,a)
        A = np.zeros(len(actions))
        self._qvalues[state][0]
        
        for i, a in enumerate(actions):
            
            A[i] = self._qvalues[state][a]
            
        # choosing the best action
        best_a = np.argmax(A)
        best_a = actions[best_a]
        limit = random.uniform(0,1)
        
        # best action if random value is superior of epsilon
        # random action if random value is inferior of epsilon

        if limit <= self.epsilon:
            choice_a = np.random.choice(len(actions))
            choice_a = actions[choice_a]
        else:
            choice_a = best_a
        return choice_a
    
           
    def train_agent(self, env, n_episodes, train=True, t_max = 10000, eps_decrease = 0.999):

        # playing the game based on gym library
        total_reward = []
        
        for i in range(n_episodes):
            s = env.reset()
            rewards_episode = 0
            self.epsilon *= eps_decrease
            
            for t in range(t_max):
            
                # get action
                a = self.get_action(s)

                # playing the game
                # getting next state, immediate reward, if game is done, info
                next_s, r , done, _ = env.step(a)

                rewards_episode += r

                # train our model based on actions played
                if train:
                    self.train_qsa(s,a,r,next_s)

                s = next_s
                if done:
                    total_reward.append(rewards_episode)
                    break
                
        return total_reward