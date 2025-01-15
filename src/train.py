import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

# ------------------------------------------------------------------------------
# Create environment and config
# ------------------------------------------------------------------------------
env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

config = {
    'state_dim': env.observation_space.shape[0],
    'nb_actions': env.action_space.n,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'buffer_size': 100000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.0,
    'epsilon_decay_period': 21000,
    'epsilon_delay_decay': 40,
    'batch_size': 500,
    'gradient_steps': 4,
    'update_target_strategy': 'replace', 
    'update_target_freq': 600,
    'criterion': torch.nn.SmoothL1Loss(),
    'model_path': "last_chance.pkl"
}

# ------------------------------------------------------------------------------
# Define DQN class
# ------------------------------------------------------------------------------
class DQN(nn.Module):
    """
    A neural network for DQN, composed of several fully-connected layers.
    """
    def __init__(self, state_dim, nb_neurons, n_action):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, 128),
            nn.ReLU(),
            nn.Linear(128, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# ReplayBuffer
# ------------------------------------------------------------------------------
class ReplayBuffer:
    """
    A simple Replay Buffer that stores past transitions (s, a, r, s', done)
    and allows for sampling random batches for training.
    """
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.data = []
        self.index = 0

    def append(self, s, a, r, s_, d):
        """Add a new transition to the buffer."""
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        batch = random.sample(self.data, batch_size)
        # Convert sampled batch to tensors on the specified device
        return list(map(
            lambda x: torch.Tensor(np.array(x)).to(self.device),
            zip(*batch)
        ))

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.data)

# ------------------------------------------------------------------------------
# ProjectAgent
# ------------------------------------------------------------------------------
class ProjectAgent:
    """
    A DQN-based agent that interacts with the environment, collects experiences,
    and updates its Q-network parameters via gradient steps on sampled replay data.
    """

    def __init__(self):
        # Environment-based dimensions
        state_dim = config['state_dim']
        n_action = config['nb_actions']

        # Build the DQN model using the class defined above
        nb_neurons = 256
        self.model = DQN(
            state_dim=state_dim,
            nb_neurons=nb_neurons,
            n_action=n_action
        )

        # Determine the device to use (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Replay buffer
        self.buffer_size = config.get('buffer_size', int(1e5))
        self.memory = ReplayBuffer(self.buffer_size, self.device)

        # DQN hyperparameters
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_gradient_steps = config['gradient_steps']

        # Exploration settings
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop

        # Target network updates
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']

        # Loss and optimizer
        self.criterion = config['criterion']
        self.lr = config['learning_rate']
        self.optimizer =  optim.Adam(self.model.parameters(), lr=self.lr)

        # Create a target model and a best model
        self.target_model = deepcopy(self.model).to(self.device)
        self.best_model = deepcopy(self.model).to(self.device)


    def act(self, observation, use_random=False):
        """
        Select an action using the current Q-network.
        Uses greedy selection (argmax) unless randomness is enforced externally.
        """
        with torch.no_grad():
            obs_tensor = torch.Tensor(observation).unsqueeze(0).to(self.device)
            q_values = self.model(obs_tensor)
            return torch.argmax(q_values).item()

    def save(self, path=None):
        """
        Save the best performing model's weights to disk.
        """
        if path is None:
            path = config['model_path']
        full_path = os.path.join(os.getcwd(), path)
        torch.save(self.best_model.state_dict(), full_path)

    def load(self):
        """
        Load a previously saved model from disk into the current Q-network.
        """
        path = config['model_path']
        full_path = os.path.join(os.getcwd(), path)
        state_dict = torch.load(full_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def gradient_step(self):
        """
        Performs a single gradient update on the Q-network using a sampled mini-batch from the replay buffer.
        """
        if len(self.memory) <= self.batch_size:
            return

        X, A, R, Y, D = self.memory.sample(self.batch_size)

        Q_next_max = self.target_model(Y).max(1)[0].detach()

        target = torch.addcmul(R, (1 - D), Q_next_max, value=self.gamma)

        Q_current = self.model(X).gather(1, A.long().unsqueeze(1))

        loss = self.criterion(Q_current, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, max_episode):
        """
        Main training loop:
          - Collect data through interaction with the environment
          - Perform gradient steps
          - Periodically update the target network
          - Evaluate and save the best model based on evaluation scores
        """
        episode_returns = []
        episode = 0
        total_reward = 0
        state, _ = env.reset()

        epsilon = self.epsilon_max
        step_count = 0
        best_eval_score = float('-inf')

        while episode < max_episode:

            if step_count > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, done, truncated, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            total_reward += reward

            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            if step_count % self.update_target_freq == 0:
              self.target_model.load_state_dict(self.model.state_dict())
            
            step_count += 1

            if done or truncated:
                episode += 1
                eval_score = evaluate_HIV(agent=self, nb_episode=1)
                eval_score_pop = evaluate_HIV_population(agent=self, nb_episode=1)
                
                print(
                    f"Episode {episode:3d} | "
                    f"Epsilon {epsilon:6.2f} | "
                    f"Batch Size {len(self.memory):5d} | "
                    f"Episode Return {total_reward:.2e} | "
                    f"Eval Score {eval_score:.2e} | "
                    f"Eval Score Pop {eval_score_pop:.2e}" 
                )

                state, _ = env.reset()

                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    self.best_model = deepcopy(self.model).to(self.device)

                if episode % 10 == 0:
                    self.save()

                episode_returns.append(total_reward)
                total_reward = 0
            else:
                state = next_state

        return episode_returns

# ------------------------------------------------------------------------------
# Main training entry point
# ------------------------------------------------------------------------------

""" if __name__ == "__main__":
    print("Training the agent...")
    agent = ProjectAgent()
    agent.train(env, max_episode=200)

    # Save the final model using the path from config
    agent.save(config['model_path'])
    print(f"Training complete. Model saved as {config['model_path']}")
 """