"""
Deep Q-Network (DQN) Agent for Dynamic Pricing
Implements DQN with experience replay and target network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Tuple, Optional


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: Tuple[int, ...] = (128, 64)):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            n_actions: Number of discrete actions
            hidden_dims: Tuple of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dims: Tuple[int, ...] = (128, 64)
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            n_actions: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_size: Replay buffer capacity
            batch_size: Mini-batch size for training
            target_update_freq: Frequency of target network updates
            hidden_dims: Hidden layer dimensions
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_revenues': [],
            'epsilon_history': [],
            'losses': [],
            'exploration_count': 0,
            'exploitation_count': 0,
            'update_count': 0
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.n_actions)
            self.training_stats['exploration_count'] += 1
        else:
            # Exploit: best action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
            self.training_stats['exploitation_count'] += 1
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Update network using experience replay.
        
        Returns:
            Loss value if update performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update statistics
        self.training_stats['update_count'] += 1
        self.training_stats['losses'].append(loss.item())
        
        # Update target network periodically
        if self.training_stats['update_count'] % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_stats['epsilon_history'].append(self.epsilon)
    
    def save(self, filepath: str):
        """Save agent to file."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
            'params': {
                'state_dim': self.state_dim,
                'n_actions': self.n_actions,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_stats = checkpoint['training_stats']
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        total_actions = (self.training_stats['exploration_count'] + 
                        self.training_stats['exploitation_count'])
        
        if total_actions > 0:
            exploration_rate = (self.training_stats['exploration_count'] / 
                              total_actions)
        else:
            exploration_rate = 0.0
        
        avg_loss = (np.mean(self.training_stats['losses'][-100:]) 
                   if self.training_stats['losses'] else 0.0)
        
        return {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'exploration_rate': exploration_rate,
            'total_updates': self.training_stats['update_count'],
            'avg_loss': avg_loss
        }
