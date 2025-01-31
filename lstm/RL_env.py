import gym
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import pickle
import csv
import sys
import torch
import time
import os

from RL_train import iterator

# Definition of hyper-parameters to tune with lower and upper value limits
# Usage -> "param" : (lower, upper)
parameters = {'dropouto' : (0.05,0.90), 'dropoute' : (0.05,0.90), 'dropouti' : (0.05,0.90), 'dropouth' : (0.05,0.90), 'wdrop' : (0.05,0.90), 'alpha' : (0.0,9.90), 'beta' : (0.0,9.90)}
# Definition of metrics presented to the agent with:
# - lower value limit
# - upper value limits
# - Thresholdout noise scale
# - for validation metrics, corresponding train metrics
# Usage -> "metric" : (lower, upper [, noise scale, train metric])
metrics = {'val_ppl' : (0.0,1000.0, 100.0, 'ppl'), 'val_loss' : (0.0,100.0, 10.0, 'loss'), 'ppl' : (0.0,1000.0), 'loss' : (0.0,100.0)}
val_metrics = {'val_ppl' : 0.0, 'val_loss' : 0.0}
not_val_metrics = {'ppl' : 0.0, 'loss' : 0.0}
# Metrics used for computing the reward of the agent with their respective coefficient factor
reward_metrics = {'val_ppl' : -100.0, 'ppl' : -1.0}

save_filename = 'normal'
corruption = 0.0


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, parameters, metrics, val_metrics, not_val_metrics, reward_metrics, save_filename, corruption, do_thresholdout=True, test_mode=False):
    super(CustomEnv, self).__init__()

    # Whether it is an environment used to train or test the agent 
    self.test_mode = test_mode
    # Whether to do thresholdout or not
    self.do_thresholdout = do_thresholdout
    # Filename used in experiments
    self.save_filename = save_filename
    # Corruption level
    self.corruption = corruption
    # RL parameters
    self.parameters = parameters
    self.metrics = metrics
    self.val_metrics = val_metrics
    self.not_val_metrics = not_val_metrics
    self.reward_metrics = reward_metrics

    # Number of train steps between changing the hyper-parameters values
    self.train_steps = 50
    # Number of validation steps used to compute validation metrics
    self.valid_steps = 1
    # Progressively allows the agent to play more time
    self.epoch_schedule = [25, 50, 75, 100]
    self.valid_batch_per_epoch = 26
    self.n_games = 0
    # Number of steps between updates of validation metrics
    # Avoids reaching the upper limit of number of thresholdout validation metrics presented to the agent
    self.steps_between_thresholdouts = 10
    self.running_avg_lgt = 10
    # Number of step statistics visible from the agent
    self.window_size = 500
    self.reset_stat()
    self.load_of_datasets_and_interaction_spaces()

    # Definition of action and observation space for the agent
    self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(len(parameters),), dtype = np.float32)
    self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(len(metrics)+1, self.window_size), dtype = np.float32)
    
  



  def load_of_datasets_and_interaction_spaces(self):
    # Define the upper and lower bounds of actions and observations
    self.lower_limit_actions = np.array([self.parameters[key][0] for key in self.parameters])
    self.upper_limit_actions = np.array([self.parameters[key][1] for key in self.parameters])

    self.lower_limit_observations = [np.ones(self.window_size)*self.metrics[key][0] for key in self.metrics]
    self.lower_limit_observations.append(np.zeros(self.window_size))
    self.lower_limit_observations = np.array(self.lower_limit_observations)

    self.upper_limit_observations = [np.ones(self.window_size)*self.metrics[key][1] for key in self.metrics]
    self.upper_limit_observations.append(np.ones(self.window_size)*100000) # Upper bound depending on the maximum number of classifier training timestep
    self.upper_limit_observations = np.array(self.upper_limit_observations)





  def action_scaling(self, action):
    # Scaling of actions from (-1, +1) range to respective (upper, lower) ranges
    range_ = self.upper_limit_actions - self.lower_limit_actions
    return (action + 1.0) * 0.5 * range_ + self.lower_limit_actions

  def observation_scaling(self, obs):
    # Verification of observed values to be in respective (upper, lower) ranges
    for i in range(len(metrics)+1):
      for j in range(self.window_size):
        if obs[i][j] > self.upper_limit_observations[i][j]:
          obs[i][j] = self.upper_limit_observations[i][j]
        if obs[i][j] < self.lower_limit_observations[i][j]:
          obs[i][j] = self.lower_limit_observations[i][j]
    # Scaling of observations from respective (upper, lower) ranges to (0, 1) range
    range_ = self.upper_limit_observations - self.lower_limit_observations
    return (obs - self.lower_limit_observations) / range_





  def step(self, hparams_scaled):
    self.n_steps += 1
    hparams = self.action_scaling(hparams_scaled)
    hparams_dict = dict(self.parameters)
    i = 0
    for key in hparams_dict:
      hparams_dict[key] = float(hparams[i])
      i += 1

    # Writing of new hyper-parameters values
    with open('%s.ser' % self.save_filename, 'wb') as fp:
      pickle.dump(hparams_dict, fp)

    try:
      stats = next(self.DNN)
      for key in self.metrics:
        for i, x in enumerate(stats[key]):
          if np.isnan(x):
            return self.observation_scaling(self.obs), 0.0, True, {}
          if x < self.metrics[key][0]:
            stats[key][i] = self.metrics[key][0]
          if x > self.metrics[key][1]:
            stats[key][i] = self.metrics[key][1]

      self.build_observation(stats)
      # Saving logs
      for key in self.metrics:
        self.logs[key].extend(stats[key])
      for key in self.parameters:
        self.logs[key].extend(stats[key])

      if not self.test_mode and self.n_games//100 < len(self.epoch_schedule) and self.n_steps//self.valid_batch_per_epoch >= self.epoch_schedule[self.n_games//100]:
        self.n_games+=1
        done = True
      else:
        done = False
    except StopIteration:
      done = True
      if (self.test_mode):
        print("This was a test ppl game for the agent")

    obs_scaled = self.observation_scaling(self.obs)
    reward = self.build_reward()

    return obs_scaled, reward, done, {}





  def reset(self):
    self.n_steps = 0
    # Initialisation of running average
    self.running_avg = {}
    for key in self.metrics:
      self.running_avg[key] = [0.0]*self.running_avg_lgt
    # Initialisation of last val ppl for the reward function
    self.last_reward_metrics = {}
    for key in self.reward_metrics:
      self.last_reward_metrics[key] = 0.0

    self.DNN = iterator(self.train_steps, self.valid_steps, self.save_filename, self.corruption)
    stats = next(self.DNN)
    self.build_observation(stats, init=True)
    self.build_reward()
    
    return self.obs



      


  def thresholdout(self, train, valid, adjustment=1.0):
    # Compute values on the standard holdout
    tolerance = 0.01*adjustment
    threshold = 0.04*adjustment

    if abs(train-valid) < threshold + np.random.normal(0,tolerance):
        valid = train
    else:
        valid += np.random.normal(0,tolerance)
    
    return valid

        

  def update_running_avg(self, stats):
    for key in self.metrics:
      if len(stats[key]) > self.running_avg_lgt:
        stats[key] = stats[key][-self.running_avg_lgt:]
    for key in self.metrics:
      for i in range(self.running_avg_lgt-len(stats[key])):
        self.running_avg[key][i] = self.running_avg[key][i+len(stats[key])]
      for i in range(len(stats[key])):
        self.running_avg[key][self.running_avg_lgt-len(stats[key])+i] = stats[key][i]



  def build_observation(self, stats, init=False):
    self.update_running_avg(stats)
    for key in self.not_val_metrics:
      self.not_val_metrics[key] = mean(self.running_avg[key])

    if self.n_steps % self.steps_between_thresholdouts == 0:
        for key in self.val_metrics:
          self.val_metrics[key] = mean(self.running_avg[key])
        if self.do_thresholdout:
          for key in self.val_metrics:
            self.val_metrics[key] = self.thresholdout(self.not_val_metrics[self.metrics[key][3]], self.val_metrics[key], self.metrics[key][2])

    # If the game begins, we initialize the obs matrix
    if init:
      self.obs = np.zeros((len(metrics)+1, self.window_size), dtype=np.float32)

    # We update it with last obtained stats
    for i in range(self.window_size-1):
      for j in range(len(self.obs)):
        self.obs[j][i] = self.obs[j][i+1]

    i = 0
    for key in self.not_val_metrics:
      self.obs[i][-1] = self.not_val_metrics[key]
      i += 1
    for key in self.val_metrics:
      self.obs[i][-1] = self.val_metrics[key]
      i += 1
    self.obs[i][-1] = mean(stats['global_step'])





  def build_reward(self):
    reward = 0.0
    valid = False
    # Calculation of validation differences
    for key in self.reward_metrics:
      if key in self.val_metrics:
        if (self.val_metrics[key] - self.last_reward_metrics[key]) == 0:
          valid = True
        reward += self.reward_metrics[key] * (self.val_metrics[key] - self.last_reward_metrics[key])
    # If validation differences are equal to zero
    # Calculation training differences
    for key in self.reward_metrics:
      if not valid and not key in self.val_metrics:
        reward += self.reward_metrics[key] * (self.not_val_metrics[key] - self.last_reward_metrics[key])
    # Update of last metrics
    for key in self.reward_metrics:
      if key in self.val_metrics:
        self.last_reward_metrics[key] = self.val_metrics[key]
      else:
        self.last_reward_metrics[key] = self.not_val_metrics[key]
    return reward
    




  def reset_stat(self):
    self.logs = {**self.metrics, **self.parameters}
    for key in self.logs:
      self.logs[key] = []

  def get_stat(self):
    return self.logs








env = CustomEnv(parameters, metrics, val_metrics, not_val_metrics, reward_metrics, save_filename, corruption)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))  

valid_env = CustomEnv(parameters, metrics, val_metrics, not_val_metrics, reward_metrics, save_filename, corruption, test_mode=True)
policy_kwargs = dict(normalize_images=False)
eval_callback = EvalCallback(valid_env, verbose=1, deterministic=True, render=False, n_eval_episodes=3, eval_freq=1000, best_model_save_path="./Agents/%s/" % save_filename)

model = SAC("MlpPolicy", env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=0)
model.learn(callback=eval_callback, total_timesteps=10000)




if not os.path.exists('Results'):
  os.makedirs('Results')
if not os.path.exists('Results/%s' % save_filename):
  os.makedirs('Results/%s' % save_filename)

logs = env.get_stat()
with open('Results/%s/train.ser' % save_filename, 'wb') as fp:
    pickle.dump(logs, fp)

logs = valid_env.get_stat()
with open('Results/%s/valid.ser' % save_filename, 'wb') as fp:
    pickle.dump(logs, fp)




env = CustomEnv(parameters, metrics, val_metrics, not_val_metrics, reward_metrics, save_filename, corruption, test_mode=True)
model = SAC.load("Agents/%s/best_model.zip" % save_filename)
for ep in range(3):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action[0])


logs = env.get_stat()
with open('Results/%s/test.ser' % save_filename, 'wb') as fp:
    pickle.dump(logs, fp)
