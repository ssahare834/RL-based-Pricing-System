"""
Market Environment for Dynamic Pricing RL System
Simulates realistic customer demand with elasticity, time variations, and competitor effects.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class MarketConfig:
    """Configuration parameters for market simulation."""
    base_demand: float = 100.0
    price_elasticity: float = -1.5  # % change in demand per % change in price
    time_variance: float = 0.3  # Amplitude of time-based demand variation
    competitor_influence: float = 0.4  # How much competitor prices affect demand
    noise_std: float = 5.0  # Random demand noise
    base_price: float = 50.0
    competitor_base_price: float = 48.0
    max_price: float = 100.0
    min_price: float = 20.0
    volatility_penalty: float = 0.05  # Penalty for price changes


class MarketEnvironment:
    """
    Simulates a dynamic pricing market environment.
    
    State space: [normalized_time, demand_level, previous_price, competitor_price]
    Action space: Discrete price points
    Reward: Revenue with penalty for price volatility
    """
    
    def __init__(self, config: Optional[MarketConfig] = None):
        self.config = config or MarketConfig()
        self.current_step = 0
        self.max_steps = 1000
        
        # Price action space (discrete)
        self.n_actions = 15
        self.price_points = np.linspace(
            self.config.min_price,
            self.config.max_price,
            self.n_actions
        )
        
        # State tracking
        self.previous_price = self.config.base_price
        self.previous_demand = self.config.base_demand
        self.history = {
            'prices': [],
            'demands': [],
            'revenues': [],
            'competitor_prices': [],
            'rewards': []
        }
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.previous_price = self.config.base_price
        self.previous_demand = self.config.base_demand
        self.history = {
            'prices': [],
            'demands': [],
            'revenues': [],
            'competitor_prices': [],
            'rewards': []
        }
        return self._get_state()
    
    def _get_time_multiplier(self) -> float:
        """Calculate time-based demand multiplier (daily/weekly patterns)."""
        # Simulate daily pattern (24-hour cycle)
        hour_of_day = (self.current_step % 24) / 24.0
        daily_pattern = 1.0 + self.config.time_variance * np.sin(2 * np.pi * hour_of_day)
        
        # Simulate weekly pattern
        day_of_week = (self.current_step % 168) / 168.0  # 168 hours in a week
        weekly_pattern = 1.0 + 0.2 * self.config.time_variance * np.sin(2 * np.pi * day_of_week)
        
        return daily_pattern * weekly_pattern
    
    def _get_competitor_price(self) -> float:
        """Simulate competitor pricing strategy."""
        # Competitor uses a simple reactive strategy with some randomness
        base = self.config.competitor_base_price
        
        # React to our previous price with delay
        if self.current_step > 10:
            avg_our_price = np.mean(self.history['prices'][-10:])
            base = 0.7 * base + 0.3 * avg_our_price
        
        # Add some random variation
        noise = np.random.normal(0, 2)
        competitor_price = np.clip(base + noise, self.config.min_price, self.config.max_price)
        
        return competitor_price
    
    def _calculate_demand(self, price: float, competitor_price: float) -> float:
        """
        Calculate demand based on price elasticity and market conditions.
        
        Demand model:
        D = D_base * time_multiplier * (P/P_base)^elasticity * competitor_factor + noise
        """
        # Time-based variation
        time_multiplier = self._get_time_multiplier()
        
        # Price elasticity effect
        price_ratio = price / self.config.base_price
        elasticity_factor = np.power(price_ratio, self.config.price_elasticity)
        
        # Competitor price effect
        price_difference = competitor_price - price
        competitor_factor = 1.0 + self.config.competitor_influence * (price_difference / self.config.base_price)
        competitor_factor = np.clip(competitor_factor, 0.5, 2.0)
        
        # Calculate base demand
        demand = (self.config.base_demand * 
                 time_multiplier * 
                 elasticity_factor * 
                 competitor_factor)
        
        # Add noise
        noise = np.random.normal(0, self.config.noise_std)
        demand = max(0, demand + noise)
        
        return demand
    
    def _calculate_reward(self, price: float, demand: float) -> float:
        """
        Calculate reward with revenue and volatility penalty.
        
        Reward = Revenue - volatility_penalty * |price_change|
        """
        revenue = price * demand
        
        # Penalize large price changes (customer confusion, brand perception)
        price_change = abs(price - self.previous_price)
        volatility_penalty = self.config.volatility_penalty * price_change * demand
        
        reward = revenue - volatility_penalty
        
        return reward
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        State: [normalized_time, demand_level, previous_price, competitor_price]
        """
        normalized_time = (self.current_step % 168) / 168.0  # Weekly cycle
        normalized_demand = self.previous_demand / (2 * self.config.base_demand)
        normalized_prev_price = self.previous_price / self.config.max_price
        
        competitor_price = self._get_competitor_price()
        normalized_competitor_price = competitor_price / self.config.max_price
        
        state = np.array([
            normalized_time,
            normalized_demand,
            normalized_prev_price,
            normalized_competitor_price
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index of price point to set
            
        Returns:
            next_state, reward, done, info
        """
        # Get price from action
        price = self.price_points[action]
        
        # Get competitor price
        competitor_price = self._get_competitor_price()
        
        # Calculate demand
        demand = self._calculate_demand(price, competitor_price)
        
        # Calculate reward
        reward = self._calculate_reward(price, demand)
        revenue = price * demand
        
        # Store history
        self.history['prices'].append(price)
        self.history['demands'].append(demand)
        self.history['revenues'].append(revenue)
        self.history['competitor_prices'].append(competitor_price)
        self.history['rewards'].append(reward)
        
        # Update state
        self.previous_price = price
        self.previous_demand = demand
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'price': price,
            'demand': demand,
            'revenue': revenue,
            'competitor_price': competitor_price,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def get_state_dim(self) -> int:
        """Return dimension of state space."""
        return 4
    
    def get_action_dim(self) -> int:
        """Return number of discrete actions."""
        return self.n_actions
