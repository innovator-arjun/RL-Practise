# -*- coding: utf-8 -*-
"""DQN-Cartpole-Pipeline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v2xN_1Kfh_SuKMD6tnd4VrBoA1Hgi14Z

# **Imports and set up**
"""

# Commented out IPython magic to ensure Python compatibility.
import gym  # Open AI gym
from torch import randint
import torch
import torch.nn as nn
import copy
import random
from collections import deque
from tqdm import tqdm


reward_arr=[]
episode_count=50  # iterations 
env=gym.make('CartPole-v0')  # creating the environment for the cartpole

"""# **The Behavior of a Random Agent**"""
print('Program started')
for i in range(episode_count):
  state=env.reset()  # reseting the environment every episode
  done=False
  score=0

  rew=0
  while(done!=True):  # to check it is achieved
    A =  randint(0,env.action_space.n,(1,))  # creating 0 or 1 options in random
    observation,reward ,  done, info =env.step(A.item())  
    rew+=reward  # doing a total sum to calculate the total reward
  reward_arr.append(rew)
  print(i,rew)

print('average reward per episode', sum(reward_arr)/len(reward_arr))

"""The agent must have some policy (aka strategy) according to which it selects an action. Just having a policy choosen random is not enough. The average reward is 25.06 out of 50 episodes the agent must have a mechanism to improve this strategy as it interacts more and more with the environment

# **DQN Agent**
"""

#Inspired from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQN_Agent:
    
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.q_net.cuda()
        self.target_net.cuda()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().cuda()
        self.experience_replay = deque(maxlen = exp_replay_size)  
        return
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.Tanh() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)
    
    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().cuda())
        Q,A = torch.max(Qp, axis=0)
        A = A if torch.rand(1,).item() > epsilon else torch.randint(0,action_space_len,(1,))
        return A
    
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q,_ = torch.max(qp, axis=1)    
        return q
    
    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return
    def load_pretrained_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.q_net.state_dict(), model_path)
    def sample_from_experience(self, sample_size):
        if(len(self.experience_replay) < sample_size):
            sample_size = len(self.experience_replay)   
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()   
        return s, a, rn, sn
    
    def train(self, batch_size ):
        s, a, rn, sn = self.sample_from_experience( sample_size = batch_size)
        if(self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        
        # predict expected return of current state using main network
        qp = self.q_net(s.cuda())
        pred_return, _ = torch.max(qp, axis=1)
        
        # get target return using target network
        q_next = self.get_q_next(sn.cuda())
        target_return = rn.cuda() + self.gamma * q_next
        
        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        self.network_sync_counter += 1       
        return loss.item()

"""# **Setting W&B**"""
print('Pushing to W&B')
wandb.login()

# 1️⃣ Start a new run, tracking config metadata
wandb.init(project="CartPole_beluga", config={
    "learning_rate":3e-3,
    "Agent": "DQN",
    "sync_freq":5
})
config = wandb.config


print('RL started')
env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256

agent = DQN_Agent(seed=1546, layer_sizes=[input_dim, 64, output_dim], lr=3e-3, sync_freq=5,
                  exp_replay_size=exp_replay_size)

"""# **Training by RL Agent to maximize the reward**"""

# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
episodes = 10000
epsilon = 1

# initiliaze experiance replay
index = 0
for i in range(exp_replay_size):
    obs = env.reset()
    done = False
    while not done:
        A = agent.get_action(obs, env.action_space.n, epsilon=1)  # geting the actions from the RL Agent
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
        obs = obs_next
        index += 1
        if index > exp_replay_size:
            break

index = 128
for i in tqdm(range(episodes)):
    obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
    while not done:
        ep_len += 1
        A = agent.get_action(obs, env.action_space.n, epsilon)  # geting the actions from the RL Agent
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])   

        obs = obs_next
        rew += reward
        index += 1

        if index > 128: # calculating loss for every 128 indexes
            index = 0
            for j in range(4):
                loss = agent.train(batch_size=16)
                losses += loss
    if epsilon > 0.05:   # epsilon annealing 
        epsilon -= (1 / 5000)
    wandb.log({"losses_list":losses / ep_len, "reward_list":rew , "episode_len_list":ep_len, 'epsilon_list':epsilon})  # to push the metrics to w&b

    losses_list.append(losses / ep_len), reward_list.append(rew)  # to append the values to the list to keep the track
    episode_len_list.append(ep_len), epsilon_list.append(epsilon) # to append the values to the list to keep the track

print("Finished!")


