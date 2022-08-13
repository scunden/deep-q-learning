#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import random
import numpy as np
from random import randint
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
SEED = 0


# In[ ]:


class Memory(object):
    def __init__(self):
        self.size = 50000
        self.memory = []
        self.batch_size = 32
        self.sample_threshold = 10000
        
    def clear_memory(self):
        if (len(self.memory) > self.size):
            self.memory.pop(0)
        else:
            pass
        
    def remember(self, observations):
        self.memory.append(observations)
        self.clear_memory()
            
    def sample(self):
        return random.sample(self.memory, self.batch_size)
    
    def unwrap_batch(self, element, mini_batch):
        if element=="s":
            return [s for s,a,r,done,s_prime in mini_batch]
        elif element=="a":
            return [a for s,a,r,done,s_prime in mini_batch]
        elif element=="r":
            return [r for s,a,r,done,s_prime in mini_batch]
        elif element=="done":
            return [done for s,a,r,done,s_prime in mini_batch]
        elif element=="s_prime":
            return [s_prime for s,a,r,done,s_prime in mini_batch]

class LLAgent(object):
    def __init__(self, states_dim, actions_dim, alpha, gamma, epsilon_decay_rate, max_episodes, verbose):
        self.verbose = verbose
        self.D = Memory()
        self.states_dim = states_dim
        self.actions_dim = actions_dim
    
        self.nodes = 64
        self.alpha = alpha
        self.Q = self.nn_model()
        self.Q_hat = keras.models.clone_model(self.Q)
        self.update_frequency = 500

        self.gamma = gamma
        self.max_epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_decay_type = "exponential"
        self.epsilon_decay_rate = epsilon_decay_rate
        
        self.max_episodes = max_episodes
        self.test_episodes = 100
        self.steps = 0
        self.reward_tracker_per_episode = []
        self.test_reward_tracker_per_episode = []
                
        
    def nn_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.states_dim )))
        model.add(Dense(self.nodes, activation='relu'))
        model.add(Dense(self.nodes, activation='relu'))
        model.add(Dense(self.actions_dim, activation='linear'))
        
        model.compile(loss='mse', optimizer="Adam")
        K.set_value(model.optimizer.learning_rate, self.alpha)
        K.set_value(model.optimizer.decay, 0)
        
        return model
    
    def update_Qhat(self):
        self.Q_hat = keras.models.clone_model(self.Q)
        self.Q_hat.set_weights(self.Q.get_weights())
        
    def epsilon_greedy_action(self, state, epsilon):         
        if (random.random() <= epsilon):
            return random.randint(0, self.actions_dim -1)
        else:
            return np.argmax(self.Q.predict(np.array([state]))[0])
     
    def decay_epsilon(self, epsilon):
        if epsilon < self.min_epsilon:
            return self.min_epsilon
        else:
            if self.epsilon_decay_type == "exponential":
                return epsilon*self.epsilon_decay_rate
            else:
                return epsilon-self.epsilon_decay_rate
         
    def update_network(self):
        
        mini_batch = self.D.sample()
        states = np.array(self.D.unwrap_batch("s",mini_batch))
        actions = self.D.unwrap_batch("a",mini_batch)
        rewards = self.D.unwrap_batch("r",mini_batch)
        dones = self.D.unwrap_batch("done",mini_batch)
        states_prime = np.array(self.D.unwrap_batch("s_prime",mini_batch))
        
        Y_s  = self.Q.predict(states)
        Y_s_prime = self.Q.predict(states_prime)
        
        Y_s_prime_Q_hat = self.Q_hat.predict(states_prime)
        Y = np.zeros((self.D.batch_size , self.actions_dim))
        
        for i in range(self.D.batch_size):
            Y[i,:] = Y_s[i]
            if dones[i]:
                Y[i, actions[i]] = rewards[i]
            else:
                Y[i, actions[i]] = rewards[i] + self.gamma*(Y_s_prime_Q_hat[i, np.argmax(Y_s_prime[i])])
                
        self.Q.fit(states, Y, batch_size=self.D.batch_size, epochs=1, verbose=False) 
        
    def train_agent(self):
        
        np.random.seed(SEED)
        random.seed(SEED)
        env = gym.make('LunarLander-v2')
        env.seed(SEED)
        
        epsilon = self.max_epsilon
        
        for episode in range(self.max_episodes):
            s = env.reset()
            done = False
            time_step = 0
            total_reward = 0.0
            epsilon = self.decay_epsilon(epsilon)
            while not done:
                self.steps += 1
                time_step += 1
                
                s = np.reshape(s, (1,self.states_dim))
                action = self.epsilon_greedy_action(s, epsilon)
                s_prime, reward, done, info = env.step(action)
                s_prime = np.reshape(s_prime, (1,self.states_dim))

                total_reward += reward
                self.D.remember((s,action,reward,done,s_prime))
                
                if len(self.D.memory) >= self.D.sample_threshold:
                    self.update_network()
                    
                    if (self.steps%self.update_frequency == 0):
                        self.update_Qhat()
                    
                s = s_prime
                
            self.reward_tracker_per_episode.append(total_reward)
            if self.verbose:
                print("Episode: {} | Steps: {} | Episode Reward: {} | Epsilon: {}".format(episode, 
                                                                                          time_step, 
                                                                                          total_reward,
                                                                                          epsilon
                                                                                         ))
                
    def test_agent(self):
        
        np.random.seed(SEED)
        random.seed(SEED)
        env = gym.make('LunarLander-v2')
        env.seed(SEED)
        
        for episode in range(self.test_episodes):
            s = env.reset()
            done = False
            time_step = 0
            total_reward = 0.0
            while not done:
                self.steps += 1
                time_step += 1
                
                s = np.reshape(s, (1,self.states_dim))
                action = np.argmax(self.Q.predict(np.array([s]))[0])
                s_prime, reward, done, info = env.step(action)
                s_prime = np.reshape(s_prime, (1,self.states_dim))

                total_reward += reward
                s = s_prime
                
            self.test_reward_tracker_per_episode.append(total_reward)
            if self.verbose:
                print("Episode: {} | Steps: {} | Episode Reward: {}".format(episode,
                                                                            time_step, 
                                                                            total_reward
                                                                           ))
                                
    def plot_train_rewards(self,plot_type="raw", y_ll=-500, y_ul=300):
        
        tracker = self.reward_tracker_per_episode
        moving_average = []
        for i in range(len(tracker)):
            moving_average.append(np.mean(tracker[:i+1][-100:]))
        if plot_type=="raw":
            toplot = tracker
            y_label = 'Score per Episode'
            figure_name = "Train Raw Score"
        else:
            toplot = moving_average
            y_label = 'Average Score over Last 100 Episodes'
            figure_name="Train Avg Score"


        plt.plot([x+1 for x in range(len(toplot))],toplot,
                 label="γ: {} | α: {} | Decay: {}".format(self.gamma,self.alpha,self.epsilon_decay_rate))

        plt.xlabel('Episode')
        plt.ylabel(y_label)
        plt.axhline(y=200, color='r')
        plt.ylim(y_ll, y_ul)

        plt.legend()
        plt.savefig('{}.png'.format(figure_name))
        print("Figure Saved")
        plt.close()
        
    def plot_test_scores(self):
        tracker = self.test_reward_tracker_per_episode

        moving_average = []
        for i in range(len(tracker)):
            moving_average.append(np.mean(tracker[:i+1][-100:]))

        plt.plot([x+1 for x in range(len(tracker))],tracker,
                 label="Raw Score".format(self.gamma,self.alpha,self.epsilon_decay_rate))
        plt.plot([x+1 for x in range(len(moving_average))],moving_average,
                 label="Moving Average".format(self.gamma,self.alpha,self.epsilon_decay_rate))

        plt.xlabel('Episode')
        plt.ylabel('Total Score')
        plt.axhline(y=200, color='r')
        plt.ylim(-100, 400)

        plt.legend()
        plt.savefig('{}.png'.format("Test Scores"))
        print("Figure Saved")
        plt.close()

class HyperparameterTuning(object):
    def __init__(self, gamma_ls, alpha_ls, decay_ls, tuning_episodes):
        self.starting_gamma = 0.99
        self.starting_alpha = 0.00025
        self.starting_epsilon_decay_rate = 0.995
        
        self.optimal_gamma = 0.99
        self.optimal_alpha = 0.001
        self.optimal_epsilon_decay_rate = 0.99
        
        self.gamma_ls = gamma_ls
        self.alpha_ls = alpha_ls
        self.decay_ls = decay_ls
        
        self.tuning_episodes = tuning_episodes

    def tune_gamma(self):
        tracker = {}
        for gamma in self.gamma_ls:
            print("Tuning Gamma {}".format(gamma))
            ll = LLAgent(states_dim=8, 
                         actions_dim=4, 
                         alpha=self.starting_alpha, 
                         gamma=gamma, 
                         epsilon_decay_rate=self.starting_epsilon_decay_rate, 
                         max_episodes=self.tuning_episodes,
                         verbose=False
                        )
            ll.train_agent()
            tracker[gamma, self.starting_alpha, self.starting_epsilon_decay_rate] = ll.reward_tracker_per_episode

        return tracker

    def tune_alpha(self):
        tracker = {}
        for alpha in self.alpha_ls:
            print("Tuning Alpha {}".format(alpha))
            ll = LLAgent(states_dim=8, 
                         actions_dim=4, 
                         alpha=alpha, 
                         gamma=self.optimal_gamma, 
                         epsilon_decay_rate=self.starting_epsilon_decay_rate, 
                         max_episodes=self.tuning_episodes,
                         verbose=False
                        )
            ll.train_agent()
            tracker[self.optimal_gamma, alpha, self.starting_epsilon_decay_rate] = ll.reward_tracker_per_episode

        return tracker

    def tune_decay(self):
        tracker = {}
        for epsilon_decay_rate in self.decay_ls:
            print("Tuning Decay {}".format(epsilon_decay_rate))
            ll = LLAgent(states_dim=8, 
                         actions_dim=4, 
                         alpha=self.optimal_alpha, 
                         gamma=self.optimal_gamma, 
                         epsilon_decay_rate=epsilon_decay_rate, 
                         max_episodes=self.tuning_episodes,
                         verbose=False
                        )
            ll.train_agent()
            tracker[self.optimal_gamma, self.optimal_alpha, epsilon_decay_rate] = ll.reward_tracker_per_episode

        return tracker

    def plot_tuning(self, tracker, figure_name):

        combination_tracker = tracker.copy()
        gammas = list(set([x[0] for x in combination_tracker.keys()]))

        alphas = list(set([x[1] for x in combination_tracker.keys()]))
        epsilons = list(set([x[2] for x in combination_tracker.keys()]))
        for gamma in gammas:
            for alpha in alphas:
                for epsilon in epsilons:

                    score_tracker = combination_tracker[gamma,alpha,epsilon]
                    moving_average = []
                    for i in range(len(score_tracker)):
                        moving_average.append(np.mean(score_tracker[:i+1][-100:]))


                    plt.plot([x+1 for x in range(len(moving_average))],moving_average,
                             label="γ: {} | α: {} | Decay: {}".format(gamma,alpha,epsilon))
                    plt.xlabel('Episode')
                    plt.ylabel('Average Score over Last 100 Episodes')
                    plt.ylim(-300, 300)
                    
                    

        plt.legend()
        plt.savefig('{}.png'.format(figure_name))
        print("Figure Saved")
        plt.close()
        


# In[ ]:


TUNE = False
if TUNE:
    tuner = HyperparameterTuning(gamma_ls=[0.9,0.99,0.999], 
                         alpha_ls=[0.01,0.001,0.00025], 
                         decay_ls=[0.9, 0.99, 0.995], 
                         tuning_episodes=350)

    gamma_tracker = tuner.tune_gamma()
    tuner.plot_tuning(gamma_tracker,"Gamma_Tuning")

    alpha_tracker = tuner.tune_alpha()
    tuner.plot_tuning(alpha_tracker,"Alpha_Tuning")

    decay_tracker = tuner.tune_decay()
    tuner.plot_tuning(decay_tracker,"Decay_Tuning")

else:
    ll = LLAgent(states_dim=8, 
         actions_dim=4, 
         alpha=0.001, 
         gamma=0.99, 
         epsilon_decay_rate=0.99, 
         max_episodes=500,
         verbose=True
        )
    ll.train_agent()
    ll.plot_train_rewards(plot_type="raw")
    ll.plot_train_rewards(plot_type="avg")

    ll.test_agent()
    ll.plot_test_scores()


# In[ ]:




