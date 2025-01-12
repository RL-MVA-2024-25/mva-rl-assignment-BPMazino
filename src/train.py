from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

config = {
    'state_dim': 6,
    'action_dim': 4,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.999,
    'batch_size': 64,
    'update_target': 10,
    'memory_size': 10000,
    'gradient_steps': 3,
}


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done
    def __len__(self):
        return len(self.buffer)

class ProjectAgent:
    def __init__(self, config, save_path="model.pth"):
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.update_target = config['update_target']
        self.memory = deque(maxlen=config['memory_size'])
        self.gradient_steps = config['gradient_steps']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.MSELoss()
        self.save_path = save_path
        self.steps = 0
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def act(self, observation, use_random=False):
        if use_random or np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
            return self.model(observation).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self):
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'target_model_state_dict': self.target_model.state_dict(),
                    'epsilon': self.epsilon,
                    }, self.save_path)

        

    def load(self):
        if  os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.model.eval()
    
def train(agent, env, nb_episodes):

    for episode in range(nb_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.steps += 1
            agent.replay()
        
        if episode % agent.update_target == 0:
            agent.update_target_network()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        agent.save()
