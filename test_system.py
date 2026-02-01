"""
Test script for RL Dynamic Pricing System
Run this to validate all components work correctly.
"""

import numpy as np
from market_environment import MarketEnvironment, MarketConfig
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from trainer import RLTrainer


def test_market_environment():
    """Test market environment functionality."""
    print("Testing Market Environment...")
    
    config = MarketConfig()
    env = MarketEnvironment(config)
    
    # Test reset
    state = env.reset()
    assert state.shape == (4,), "State shape incorrect"
    print("✓ Environment reset works")
    
    # Test step
    action = 7  # Middle price point
    next_state, reward, done, info = env.step(action)
    
    assert next_state.shape == (4,), "Next state shape incorrect"
    assert isinstance(reward, float), "Reward should be float"
    assert isinstance(done, bool), "Done should be boolean"
    assert 'price' in info, "Info should contain price"
    assert 'demand' in info, "Info should contain demand"
    print("✓ Environment step works")
    
    # Test episode
    state = env.reset()
    for _ in range(100):
        action = np.random.randint(env.get_action_dim())
        next_state, reward, done, info = env.step(action)
        if done:
            break
        state = next_state
    
    assert len(env.history['prices']) == 100, "History tracking failed"
    print("✓ Full episode works")
    print(f"  - Average price: ${np.mean(env.history['prices']):.2f}")
    print(f"  - Average demand: {np.mean(env.history['demands']):.1f}")
    print(f"  - Total revenue: ${np.sum(env.history['revenues']):,.0f}")
    

def test_qlearning_agent():
    """Test Q-Learning agent."""
    print("\nTesting Q-Learning Agent...")
    
    agent = QLearningAgent(n_actions=15, state_dim=4)
    
    # Test action selection
    state = np.random.rand(4)
    action = agent.select_action(state)
    assert 0 <= action < 15, "Action out of bounds"
    print("✓ Action selection works")
    
    # Test update
    next_state = np.random.rand(4)
    td_error = agent.update(state, action, 100.0, next_state, False)
    assert isinstance(td_error, float), "TD error should be float"
    print("✓ Q-value update works")
    
    # Test epsilon decay
    initial_epsilon = agent.epsilon
    agent.decay_epsilon()
    assert agent.epsilon < initial_epsilon, "Epsilon should decrease"
    print("✓ Epsilon decay works")


def test_dqn_agent():
    """Test DQN agent."""
    print("\nTesting DQN Agent...")
    
    agent = DQNAgent(state_dim=4, n_actions=15)
    
    # Test action selection
    state = np.random.rand(4)
    action = agent.select_action(state)
    assert 0 <= action < 15, "Action out of bounds"
    print("✓ Action selection works")
    
    # Test storing transitions
    next_state = np.random.rand(4)
    agent.store_transition(state, action, 100.0, next_state, False)
    assert len(agent.replay_buffer) == 1, "Transition not stored"
    print("✓ Transition storage works")
    
    # Fill buffer
    for _ in range(100):
        s = np.random.rand(4)
        a = np.random.randint(15)
        r = np.random.rand() * 100
        ns = np.random.rand(4)
        agent.store_transition(s, a, r, ns, False)
    
    # Test update
    loss = agent.update()
    assert loss is not None, "Update should return loss"
    print("✓ Network update works")


def test_training():
    """Test training pipeline."""
    print("\nTesting Training Pipeline...")
    
    # Test Q-Learning training
    print("Training Q-Learning for 10 episodes...")
    qlearning_trainer = RLTrainer('qlearning')
    history = qlearning_trainer.train(10, verbose=False)
    
    assert len(history['episode_rewards']) == 10, "History length incorrect"
    print("✓ Q-Learning training works")
    print(f"  - Final episode reward: {history['episode_rewards'][-1]:.0f}")
    print(f"  - Final episode revenue: ${history['episode_revenues'][-1]:,.0f}")
    
    # Test DQN training
    print("Training DQN for 10 episodes...")
    dqn_trainer = RLTrainer('dqn')
    history = dqn_trainer.train(10, verbose=False)
    
    assert len(history['episode_rewards']) == 10, "History length incorrect"
    print("✓ DQN training works")
    print(f"  - Final episode reward: {history['episode_rewards'][-1]:.0f}")
    print(f"  - Final episode revenue: ${history['episode_revenues'][-1]:,.0f}")


def test_evaluation():
    """Test evaluation functionality."""
    print("\nTesting Evaluation...")
    
    trainer = RLTrainer('qlearning')
    trainer.train(20, verbose=False)
    
    # Evaluate
    eval_metrics = trainer.evaluate(5)
    
    assert 'mean_reward' in eval_metrics, "Missing mean_reward"
    assert 'mean_revenue' in eval_metrics, "Missing mean_revenue"
    print("✓ Evaluation works")
    print(f"  - Mean reward: {eval_metrics['mean_reward']:.0f}")
    print(f"  - Mean revenue: ${eval_metrics['mean_revenue']:,.0f}")
    
    # Comparison
    comparison = trainer.compare_with_baseline('fixed', 5)
    
    assert 'rl_metrics' in comparison, "Missing RL metrics"
    assert 'baseline_metrics' in comparison, "Missing baseline metrics"
    assert 'revenue_improvement_pct' in comparison, "Missing improvement metric"
    print("✓ Baseline comparison works")
    print(f"  - RL revenue: ${comparison['rl_metrics']['mean_revenue']:,.0f}")
    print(f"  - Baseline revenue: ${comparison['baseline_metrics']['mean_revenue']:,.0f}")
    print(f"  - Improvement: {comparison['revenue_improvement_pct']:.1f}%")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RL DYNAMIC PRICING SYSTEM - VALIDATION TESTS")
    print("=" * 60)
    
    try:
        test_market_environment()
        test_qlearning_agent()
        test_dqn_agent()
        test_training()
        test_evaluation()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run: streamlit run app.py")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()
