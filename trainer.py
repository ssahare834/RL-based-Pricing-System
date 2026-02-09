

import numpy as np
from typing import Dict, List, Optional, Tuple
from market_environment import MarketEnvironment, MarketConfig
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent


class BaselinePricer:
    """Static baseline pricing strategies for comparison."""
    
    def __init__(self, strategy: str = 'fixed', base_price: float = 50.0):
        """
        Initialize baseline pricer.
        
        Args:
            strategy: 'fixed', 'random', or 'simple_rule'
            base_price: Base price for strategies
        """
        self.strategy = strategy
        self.base_price = base_price
    
    def get_price(self, state: np.ndarray, price_points: np.ndarray) -> float:
        """Get price based on strategy."""
        if self.strategy == 'fixed':
            return self.base_price
        
        elif self.strategy == 'random':
            return np.random.choice(price_points)
        
        elif self.strategy == 'simple_rule':
            # Simple rule: adjust based on demand
            demand_level = state[1]  # Normalized demand
            if demand_level > 0.7:
                return self.base_price * 1.1
            elif demand_level < 0.3:
                return self.base_price * 0.9
            else:
                return self.base_price
        
        return self.base_price


class RLTrainer:
    """Trainer for RL agents in pricing environment."""
    
    def __init__(self, agent_type: str = 'dqn', market_config: Optional[MarketConfig] = None):
        """
        Initialize trainer.
        
        Args:
            agent_type: 'qlearning' or 'dqn'
            market_config: Market configuration
        """
        self.agent_type = agent_type
        self.env = MarketEnvironment(market_config)
        
        # Initialize agent
        if agent_type == 'qlearning':
            self.agent = QLearningAgent(
                n_actions=self.env.get_action_dim(),
                state_dim=self.env.get_state_dim(),
                learning_rate=0.1,
                gamma=0.95,
                epsilon=1.0,
                epsilon_decay=0.995
            )
        elif agent_type == 'dqn':
            self.agent = DQNAgent(
                state_dim=self.env.get_state_dim(),
                n_actions=self.env.get_action_dim(),
                learning_rate=0.001,
                gamma=0.95,
                epsilon=1.0,
                epsilon_decay=0.995
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.training_history = {
            'episode_rewards': [],
            'episode_revenues': [],
            'episode_avg_prices': [],
            'episode_avg_demands': [],
            'epsilon_history': []
        }
    
    def train_episode(self) -> Dict:
        """
        Train for one episode.
        
        Returns:
            Episode statistics
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_revenue = 0.0
        done = False
        
        while not done:
            # Select and execute action
            action = self.agent.select_action(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            
            # Store/update
            if self.agent_type == 'dqn':
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.update()
            else:  # qlearning
                self.agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_revenue += info['revenue']
            state = next_state
        
        # Decay epsilon
        self.agent.decay_epsilon()
        
        # Episode statistics
        episode_stats = {
            'reward': episode_reward,
            'revenue': episode_revenue,
            'avg_price': np.mean(self.env.history['prices']),
            'avg_demand': np.mean(self.env.history['demands']),
            'epsilon': self.agent.epsilon
        }
        
        return episode_stats
    
    def train(self, n_episodes: int, verbose: bool = True) -> Dict:
        """
        Train agent for multiple episodes.
        
        Args:
            n_episodes: Number of episodes to train
            verbose: Whether to show progress
            
        Returns:
            Training history
        """
        for episode in range(n_episodes):
            stats = self.train_episode()
            
            # Store history
            self.training_history['episode_rewards'].append(stats['reward'])
            self.training_history['episode_revenues'].append(stats['revenue'])
            self.training_history['episode_avg_prices'].append(stats['avg_price'])
            self.training_history['episode_avg_demands'].append(stats['avg_demand'])
            self.training_history['epsilon_history'].append(stats['epsilon'])
            
            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_revenue = np.mean(self.training_history['episode_revenues'][-10:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Revenue: ${avg_revenue:.2f} | "
                      f"Epsilon: {stats['epsilon']:.3f}")
        
        if verbose:
            print(f"\n✅ Training complete! Final metrics:")
            print(f"   Avg Reward (last 20): {np.mean(self.training_history['episode_rewards'][-20:]):.2f}")
            print(f"   Avg Revenue (last 20): ${np.mean(self.training_history['episode_revenues'][-20:]):.2f}")
        
        return self.training_history
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        Evaluate trained agent.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        eval_rewards = []
        eval_revenues = []
        eval_prices = []
        eval_demands = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_revenue = 0.0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_revenue += info['revenue']
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_revenues.append(episode_revenue)
            eval_prices.append(np.mean(self.env.history['prices']))
            eval_demands.append(np.mean(self.env.history['demands']))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_revenue': np.mean(eval_revenues),
            'std_revenue': np.std(eval_revenues),
            'mean_price': np.mean(eval_prices),
            'mean_demand': np.mean(eval_demands)
        }
    
    def compare_with_baseline(
        self,
        baseline_strategy: str = 'fixed',
        n_episodes: int = 10
    ) -> Dict:
        """
        Compare RL agent with baseline strategy.
        
        Args:
            baseline_strategy: Baseline strategy to compare
            n_episodes: Number of comparison episodes
            
        Returns:
            Comparison metrics
        """
        baseline = BaselinePricer(strategy=baseline_strategy, 
                                 base_price=self.env.config.base_price)
        
        # Evaluate RL agent
        rl_metrics = self.evaluate(n_episodes)
        
        # Evaluate baseline
        baseline_rewards = []
        baseline_revenues = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_revenue = 0.0
            done = False
            
            while not done:
                price = baseline.get_price(state, self.env.price_points)
                action = np.argmin(np.abs(self.env.price_points - price))
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_revenue += info['revenue']
                state = next_state
            
            baseline_rewards.append(episode_reward)
            baseline_revenues.append(episode_revenue)
        
        baseline_metrics = {
            'mean_reward': np.mean(baseline_rewards),
            'mean_revenue': np.mean(baseline_revenues)
        }
        
        # Calculate improvement
        reward_improvement = ((rl_metrics['mean_reward'] - baseline_metrics['mean_reward']) / 
                             abs(baseline_metrics['mean_reward']) * 100)
        revenue_improvement = ((rl_metrics['mean_revenue'] - baseline_metrics['mean_revenue']) / 
                              baseline_metrics['mean_revenue'] * 100)
        
        return {
            'rl_metrics': rl_metrics,
            'baseline_metrics': baseline_metrics,
            'reward_improvement_pct': reward_improvement,
            'revenue_improvement_pct': revenue_improvement
        }


def run_comparison_experiment(
    n_episodes: int = 200,
    market_config: Optional[MarketConfig] = None
) -> Dict:
    """
    Run comparison experiment between Q-Learning, DQN, and baseline.
    
    Args:
        n_episodes: Number of training episodes
        market_config: Market configuration
        
    Returns:
        Results for all methods
    """
    results = {}
    
    # Train Q-Learning
    print("Training Q-Learning agent...")
    qlearning_trainer = RLTrainer('qlearning', market_config)
    qlearning_trainer.train(n_episodes, verbose=True)
    results['qlearning'] = {
        'training_history': qlearning_trainer.training_history,
        'evaluation': qlearning_trainer.evaluate(20),
        'comparison': qlearning_trainer.compare_with_baseline('fixed', 20)
    }
    
    # Train DQN
    print("\nTraining DQN agent...")
    dqn_trainer = RLTrainer('dqn', market_config)
    dqn_trainer.train(n_episodes, verbose=True)
    results['dqn'] = {
        'training_history': dqn_trainer.training_history,
        'evaluation': dqn_trainer.evaluate(20),
        'comparison': dqn_trainer.compare_with_baseline('fixed', 20)
    }
    
    return results
