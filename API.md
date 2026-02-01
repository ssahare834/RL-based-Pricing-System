# API Documentation

## Overview

This document describes the internal APIs and interfaces of the RL Dynamic Pricing System.

## Core Components

### 1. MarketEnvironment

**Purpose**: Simulates market dynamics for pricing decisions

**Initialization**:
```python
from market_environment import MarketEnvironment, MarketConfig

config = MarketConfig(
    base_demand=100.0,
    price_elasticity=-1.5,
    time_variance=0.3,
    competitor_influence=0.4,
    noise_std=5.0
)

env = MarketEnvironment(config)
```

**Methods**:

#### `reset() -> np.ndarray`
Resets environment to initial state.

**Returns**:
- `state`: 4D numpy array [normalized_time, demand_level, previous_price, competitor_price]

**Example**:
```python
state = env.reset()
# state: array([0.5, 0.6, 0.5, 0.48])
```

#### `step(action: int) -> Tuple[np.ndarray, float, bool, Dict]`
Executes one timestep.

**Parameters**:
- `action`: Index of price point (0-14)

**Returns**:
- `next_state`: Next state (4D array)
- `reward`: Immediate reward (float)
- `done`: Whether episode terminated (bool)
- `info`: Additional information (dict)

**Example**:
```python
action = 7  # Mid-range price
next_state, reward, done, info = env.step(action)

# info contains:
# {
#   'price': 55.0,
#   'demand': 98.5,
#   'revenue': 5417.5,
#   'competitor_price': 48.2,
#   'step': 1
# }
```

#### `get_state_dim() -> int`
Returns state space dimension (always 4).

#### `get_action_dim() -> int`
Returns number of discrete actions (default 15).

**Properties**:
- `price_points`: Array of available price points
- `history`: Dictionary containing episode history
  - `prices`: List of prices set
  - `demands`: List of observed demands
  - `revenues`: List of revenues earned
  - `competitor_prices`: List of competitor prices
  - `rewards`: List of rewards received

---

### 2. QLearningAgent

**Purpose**: Tabular Q-learning agent

**Initialization**:
```python
from q_learning_agent import QLearningAgent

agent = QLearningAgent(
    n_actions=15,
    state_dim=4,
    learning_rate=0.1,
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    n_state_bins=10
)
```

**Methods**:

#### `select_action(state: np.ndarray, training: bool = True) -> int`
Selects action using epsilon-greedy policy.

**Parameters**:
- `state`: Current state (4D array)
- `training`: Whether in training mode (enables exploration)

**Returns**:
- `action`: Selected action index

**Example**:
```python
state = env.reset()
action = agent.select_action(state, training=True)
# action: 7 (with exploration) or best_action (exploitation)
```

#### `update(state, action, reward, next_state, done) -> float`
Updates Q-values using Q-learning rule.

**Parameters**:
- `state`: Current state
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Next state
- `done`: Episode termination flag

**Returns**:
- `td_error`: Temporal difference error

**Example**:
```python
td_error = agent.update(state, action, reward, next_state, False)
# td_error: 15.3 (magnitude of learning)
```

#### `decay_epsilon()`
Decays exploration rate.

**Example**:
```python
agent.decay_epsilon()
# epsilon: 1.0 -> 0.995
```

#### `save(filepath: str)`
Saves agent to file.

#### `load(filepath: str)`
Loads agent from file.

#### `get_stats() -> Dict`
Returns training statistics.

**Returns**:
```python
{
    'epsilon': 0.1,
    'q_table_size': 1523,
    'exploration_rate': 0.15,
    'total_updates': 10000
}
```

---

### 3. DQNAgent

**Purpose**: Deep Q-Network agent with neural network approximation

**Initialization**:
```python
from dqn_agent import DQNAgent

agent = DQNAgent(
    state_dim=4,
    n_actions=15,
    learning_rate=0.001,
    gamma=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    target_update_freq=10,
    hidden_dims=(128, 64)
)
```

**Methods**:

#### `select_action(state: np.ndarray, training: bool = True) -> int`
Selects action using neural network.

**Example**:
```python
action = agent.select_action(state, training=True)
```

#### `store_transition(state, action, reward, next_state, done)`
Stores experience in replay buffer.

**Example**:
```python
agent.store_transition(state, action, reward, next_state, False)
```

#### `update() -> Optional[float]`
Performs mini-batch gradient descent.

**Returns**:
- `loss`: Training loss (or None if buffer too small)

**Example**:
```python
loss = agent.update()
# loss: 0.023 (mean squared error)
```

#### `save(filepath: str)`
Saves model weights and optimizer state.

#### `load(filepath: str)`
Loads model weights and optimizer state.

#### `get_stats() -> Dict`
Returns training statistics.

**Returns**:
```python
{
    'epsilon': 0.1,
    'buffer_size': 9834,
    'exploration_rate': 0.12,
    'total_updates': 5432,
    'avg_loss': 0.015
}
```

---

### 4. RLTrainer

**Purpose**: Training pipeline for RL agents

**Initialization**:
```python
from trainer import RLTrainer

trainer = RLTrainer(
    agent_type='dqn',  # or 'qlearning'
    market_config=config
)
```

**Methods**:

#### `train(n_episodes: int, verbose: bool = True) -> Dict`
Trains agent for specified episodes.

**Parameters**:
- `n_episodes`: Number of training episodes
- `verbose`: Show progress bar

**Returns**:
- Training history dictionary

**Example**:
```python
history = trainer.train(n_episodes=200, verbose=True)

# history contains:
# {
#   'episode_rewards': [...],
#   'episode_revenues': [...],
#   'episode_avg_prices': [...],
#   'episode_avg_demands': [...],
#   'epsilon_history': [...]
# }
```

#### `evaluate(n_episodes: int = 10) -> Dict`
Evaluates trained agent.

**Returns**:
```python
{
    'mean_reward': 5234.2,
    'std_reward': 234.5,
    'mean_revenue': 48392.1,
    'std_revenue': 1892.3,
    'mean_price': 52.3,
    'mean_demand': 95.2
}
```

#### `compare_with_baseline(baseline_strategy: str, n_episodes: int) -> Dict`
Compares agent with baseline strategy.

**Parameters**:
- `baseline_strategy`: 'fixed', 'random', or 'simple_rule'
- `n_episodes`: Number of comparison episodes

**Returns**:
```python
{
    'rl_metrics': {...},
    'baseline_metrics': {...},
    'reward_improvement_pct': 18.5,
    'revenue_improvement_pct': 21.3
}
```

---

### 5. BaselinePricer

**Purpose**: Static baseline pricing strategies

**Initialization**:
```python
from trainer import BaselinePricer

baseline = BaselinePricer(
    strategy='fixed',  # or 'random', 'simple_rule'
    base_price=50.0
)
```

**Methods**:

#### `get_price(state: np.ndarray, price_points: np.ndarray) -> float`
Returns price based on strategy.

**Strategies**:
- `'fixed'`: Always returns base_price
- `'random'`: Random price from price_points
- `'simple_rule'`: Adjusts based on demand level

---

## Usage Examples

### Complete Training Pipeline

```python
from market_environment import MarketConfig
from trainer import RLTrainer

# Configure market
config = MarketConfig(
    base_demand=100.0,
    price_elasticity=-1.5,
    time_variance=0.3,
    competitor_influence=0.4
)

# Create trainer
trainer = RLTrainer('dqn', config)

# Train
history = trainer.train(n_episodes=200)

# Evaluate
evaluation = trainer.evaluate(n_episodes=20)
print(f"Mean Revenue: ${evaluation['mean_revenue']:,.0f}")

# Compare with baseline
comparison = trainer.compare_with_baseline('fixed', 20)
print(f"Improvement: {comparison['revenue_improvement_pct']:.1f}%")
```

### Custom Market Simulation

```python
from market_environment import MarketEnvironment, MarketConfig

# Create custom market
config = MarketConfig(
    base_demand=150.0,
    price_elasticity=-2.0,  # More elastic
    time_variance=0.5,      # Higher variation
    competitor_influence=0.6 # Strong competition
)

env = MarketEnvironment(config)

# Simulate episode
state = env.reset()
total_revenue = 0

for step in range(100):
    # Manual pricing strategy
    if state[1] > 0.7:  # High demand
        price = 60.0
    else:
        price = 45.0
    
    action = np.argmin(np.abs(env.price_points - price))
    next_state, reward, done, info = env.step(action)
    
    total_revenue += info['revenue']
    state = next_state

print(f"Total Revenue: ${total_revenue:,.0f}")
```

### Agent Comparison

```python
from trainer import run_comparison_experiment

# Run full comparison
results = run_comparison_experiment(
    n_episodes=200,
    market_config=config
)

# Results for all methods
for method in ['qlearning', 'dqn']:
    metrics = results[method]['evaluation']
    comparison = results[method]['comparison']
    
    print(f"\n{method.upper()}:")
    print(f"  Revenue: ${metrics['mean_revenue']:,.0f}")
    print(f"  Improvement: {comparison['revenue_improvement_pct']:.1f}%")
```

---

## State Space Details

### State Vector Components

1. **Normalized Time** (0-1)
   - Represents position in weekly cycle
   - Captures daily/weekly demand patterns
   - Formula: `(step % 168) / 168`

2. **Demand Level** (0-2)
   - Previous demand normalized by base demand
   - Indicates market conditions
   - Formula: `previous_demand / (2 * base_demand)`

3. **Previous Price** (0-1)
   - Last price normalized by max price
   - Provides pricing history
   - Formula: `previous_price / max_price`

4. **Competitor Price** (0-1)
   - Current competitor price normalized
   - Enables competitive positioning
   - Formula: `competitor_price / max_price`

---

## Action Space Details

**Discrete Price Points**: 15 levels

```python
price_points = linspace(min_price, max_price, 15)
# [20.0, 25.71, 31.43, 37.14, 42.86, 48.57, 54.29, 
#  60.0, 65.71, 71.43, 77.14, 82.86, 88.57, 94.29, 100.0]
```

**Action Selection**:
```python
action = 7  # Selects price_points[7] = $60
```

---

## Reward Function Details

### Formula

```
R(s,a) = Price × Demand - λ × |ΔPrice| × Demand

where:
  Price = Selected price
  Demand = Realized demand
  λ = Volatility penalty (default 0.05)
  ΔPrice = |Price - Previous_Price|
```

### Components

1. **Revenue**: `Price × Demand`
   - Direct revenue from sales
   - Primary optimization objective

2. **Volatility Penalty**: `λ × |ΔPrice| × Demand`
   - Penalizes large price changes
   - Maintains customer trust
   - Weighted by demand impact

### Tuning λ

- `λ = 0`: Pure revenue maximization (volatile)
- `λ = 0.05`: Balanced (default)
- `λ = 0.1`: Stable pricing (conservative)

---

## Extension Points

### Custom Reward Functions

```python
def custom_reward(price, demand, prev_price, config):
    revenue = price * demand
    
    # Add custom penalties/bonuses
    margin_bonus = (price - 30) * demand  # Profit margin
    volatility_penalty = config.volatility_penalty * abs(price - prev_price)
    
    return revenue + margin_bonus - volatility_penalty
```

### Custom Market Dynamics

```python
class CustomEnvironment(MarketEnvironment):
    def _calculate_demand(self, price, competitor_price):
        # Override with custom demand model
        base_demand = super()._calculate_demand(price, competitor_price)
        
        # Add seasonal effects
        season_multiplier = self._get_seasonal_effect()
        
        return base_demand * season_multiplier
```

---

## Performance Benchmarks

### Training Times (200 episodes)

- **Q-Learning**: ~30 seconds
- **DQN**: ~2 minutes

### Memory Usage

- **Q-Learning**: ~100 MB
- **DQN**: ~500 MB

### Inference Speed

- **Q-Learning**: <1ms per action
- **DQN**: <10ms per action

---

## Error Handling

All methods include proper error handling:

```python
try:
    result = agent.select_action(state)
except Exception as e:
    logger.error(f"Action selection failed: {e}")
    # Fallback to default action
    result = default_action
```

---

## Testing

Run comprehensive tests:

```bash
python test_system.py
```

Test individual components:

```python
from market_environment import MarketEnvironment

env = MarketEnvironment()
assert env.get_state_dim() == 4
assert env.get_action_dim() == 15
```

---

## Version History

- **v1.0.0**: Initial release
  - Q-Learning agent
  - DQN agent
  - Market environment
  - Streamlit dashboard

---

For more information, see the main [README.md](README.md).
