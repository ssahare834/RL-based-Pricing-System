"""
Q-Learning Agent for Dynamic Pricing
Implements tabular Q-learning with discrete state-action space.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration.
    
    Uses state discretization for tabular Q-learning.
    """
    
    def __init__(
        self,
        n_actions: int,
        state_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        n_state_bins: int = 10
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_actions: Number of discrete actions
            state_dim: Dimension of state space
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            n_state_bins: Number of bins for state discretization
        """
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_state_bins = n_state_bins
        
        # Q-table: dictionary mapping (discretized_state, action) -> Q-value
        self.q_table: Dict[Tuple, np.ndarray] = {}
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_revenues': [],
            'epsilon_history': [],
            'exploration_count': 0,
            'exploitation_count': 0
        }
        
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize continuous state into bins.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple of discretized state indices
        """
        discretized = []
        for i, s in enumerate(state):
            # Clip to [0, 1] range and discretize
            s_clipped = np.clip(s, 0, 1)
            bin_idx = int(s_clipped * (self.n_state_bins - 1))
            discretized.append(bin_idx)
        
        return tuple(discretized)
    
    def _get_q_values(self, state_key: Tuple) -> np.ndarray:
        """
        Get Q-values for a state, initializing if necessary.
        
        Args:
            state_key: Discretized state tuple
            
        Returns:
            Array of Q-values for all actions
        """
        if state_key not in self.q_table:
            # Initialize with small random values
            self.q_table[state_key] = np.random.uniform(0, 0.01, self.n_actions)
        
        return self.q_table[state_key]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        state_key = self._discretize_state(state)
        q_values = self._get_q_values(state_key)
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(self.n_actions)
            self.training_stats['exploration_count'] += 1
        else:
            # Exploit: best action
            action = np.argmax(q_values)
            self.training_stats['exploitation_count'] += 1
        
        return action
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> float:
        """
        Update Q-values using Q-learning update rule.
        
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            TD error (for monitoring)
        """
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Get current Q-value
        q_values = self._get_q_values(state_key)
        current_q = q_values[action]
        
        # Get max Q-value for next state
        if done:
            target_q = reward
        else:
            next_q_values = self._get_q_values(next_state_key)
            target_q = reward + self.gamma * np.max(next_q_values)
        
        # Calculate TD error
        td_error = target_q - current_q
        
        # Update Q-value
        q_values[action] += self.learning_rate * td_error
        
        return abs(td_error)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_stats['epsilon_history'].append(self.epsilon)
    
    def save(self, filepath: str):
        """Save agent to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'training_stats': self.training_stats,
                'params': {
                    'n_actions': self.n_actions,
                    'state_dim': self.state_dim,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon_min': self.epsilon_min,
                    'epsilon_decay': self.epsilon_decay,
                    'n_state_bins': self.n_state_bins
                }
            }, f)
    
    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.training_stats = data['training_stats']
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        total_actions = (self.training_stats['exploration_count'] + 
                        self.training_stats['exploitation_count'])
        
        if total_actions > 0:
            exploration_rate = (self.training_stats['exploration_count'] / 
                              total_actions)
        else:
            exploration_rate = 0.0
        
        return {
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'exploration_rate': exploration_rate,
            'total_updates': total_actions
        }
