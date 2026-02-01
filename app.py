"""
Streamlit Dashboard for RL-Based Dynamic Pricing System
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os

from market_environment import MarketEnvironment, MarketConfig
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from trainer import RLTrainer, BaselinePricer

# Page configuration
st.set_page_config(
    page_title="RL Dynamic Pricing System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'current_env' not in st.session_state:
        st.session_state.current_env = None
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False


def create_market_config_ui():
    """Create UI for market configuration."""
    st.sidebar.header("🏪 Market Configuration")
    
    with st.sidebar.expander("Market Parameters", expanded=True):
        base_demand = st.slider("Base Demand", 50, 200, 100, 10,
                               help="Average customer demand")
        price_elasticity = st.slider("Price Elasticity", -3.0, -0.5, -1.5, 0.1,
                                     help="Demand sensitivity to price changes")
        time_variance = st.slider("Time Variance", 0.0, 1.0, 0.3, 0.05,
                                 help="Amplitude of time-based demand fluctuation")
        competitor_influence = st.slider("Competitor Influence", 0.0, 1.0, 0.4, 0.05,
                                        help="Impact of competitor prices")
        noise_std = st.slider("Demand Noise", 0.0, 20.0, 5.0, 1.0,
                             help="Random demand variation")
    
    config = MarketConfig(
        base_demand=base_demand,
        price_elasticity=price_elasticity,
        time_variance=time_variance,
        competitor_influence=competitor_influence,
        noise_std=noise_std
    )
    
    return config


def create_training_ui():
    """Create UI for model training."""
    st.header("🎓 Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        agent_type = st.selectbox(
            "Select RL Algorithm",
            ["Q-Learning", "Deep Q-Network (DQN)"],
            help="Choose the reinforcement learning algorithm"
        )
        
    with col2:
        n_episodes = st.number_input(
            "Training Episodes",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Number of episodes for training"
        )
    
    # Advanced training parameters
    with st.expander("Advanced Training Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = st.number_input("Learning Rate", 0.001, 0.5, 0.1, 0.01)
        with col2:
            gamma = st.slider("Discount Factor (γ)", 0.8, 0.99, 0.95, 0.01)
        with col3:
            epsilon_decay = st.slider("Epsilon Decay", 0.9, 0.999, 0.995, 0.001)
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        agent_key = agent_type.lower().replace(" ", "_").replace("(", "").replace(")", "")
        
        with st.spinner(f"Training {agent_type} agent..."):
            # Create trainer
            trainer = RLTrainer(
                agent_type='qlearning' if 'Q-Learning' in agent_type else 'dqn',
                market_config=st.session_state.market_config
            )
            
            # Customize agent parameters
            if hasattr(trainer.agent, 'learning_rate'):
                trainer.agent.learning_rate = learning_rate
            trainer.agent.gamma = gamma
            trainer.agent.epsilon_decay = epsilon_decay
            
            # Train
            history = trainer.train(n_episodes, verbose=False)
            
            # Evaluate
            evaluation = trainer.evaluate(20)
            comparison = trainer.compare_with_baseline('fixed', 20)
            
            # Store results
            st.session_state.trained_models[agent_key] = {
                'trainer': trainer,
                'history': history,
                'evaluation': evaluation,
                'comparison': comparison
            }
            st.session_state.training_complete = True
        
        st.success(f"✅ {agent_type} training complete!")
        st.rerun()


def plot_training_progress(history: dict, agent_name: str):
    """Plot training progress."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Episode Revenues', 
                       'Average Prices', 'Exploration Rate'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    episodes = list(range(len(history['episode_rewards'])))
    
    # Rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=history['episode_rewards'],
                  mode='lines', name='Reward', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Revenues
    fig.add_trace(
        go.Scatter(x=episodes, y=history['episode_revenues'],
                  mode='lines', name='Revenue', line=dict(color='#2ca02c')),
        row=1, col=2
    )
    
    # Prices
    fig.add_trace(
        go.Scatter(x=episodes, y=history['episode_avg_prices'],
                  mode='lines', name='Avg Price', line=dict(color='#ff7f0e')),
        row=2, col=1
    )
    
    # Epsilon
    fig.add_trace(
        go.Scatter(x=episodes, y=history['epsilon_history'],
                  mode='lines', name='Epsilon', line=dict(color='#d62728')),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=2)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="ε", row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f"{agent_name} Training Progress",
        title_x=0.5
    )
    
    return fig


def plot_comparison(models: dict):
    """Plot comparison between models and baseline."""
    model_names = []
    revenues = []
    improvements = []
    
    for name, data in models.items():
        model_names.append(name.replace('_', ' ').title())
        revenues.append(data['evaluation']['mean_revenue'])
        improvements.append(data['comparison']['revenue_improvement_pct'])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Revenue Comparison', 'Improvement vs Baseline (%)'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Revenue comparison
    fig.add_trace(
        go.Bar(x=model_names, y=revenues, 
               marker_color=['#1f77b4', '#ff7f0e'],
               text=[f'${r:,.0f}' for r in revenues],
               textposition='outside'),
        row=1, col=1
    )
    
    # Improvement comparison
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
    fig.add_trace(
        go.Bar(x=model_names, y=improvements,
               marker_color=colors,
               text=[f'{i:.1f}%' for i in improvements],
               textposition='outside'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
    fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Model Performance Comparison",
        title_x=0.5
    )
    
    return fig


def create_live_simulation():
    """Create live pricing simulation."""
    st.header("🔴 Live Pricing Simulation")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ Please train at least one model first!")
        return
    
    # Model selection
    model_options = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox(
        "Select Model for Simulation",
        [m.replace('_', ' ').title() for m in model_options]
    )
    
    model_key = selected_model.lower().replace(' ', '_')
    trainer = st.session_state.trained_models[model_key]['trainer']
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Simulation Controls")
        simulation_steps = st.slider("Simulation Steps", 50, 500, 200, 50)
        
        if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
            st.session_state.simulation_running = True
    
    if st.session_state.simulation_running:
        # Run simulation
        state = trainer.env.reset()
        done = False
        step = 0
        
        # Containers for real-time updates
        metrics_container = st.container()
        chart_container = st.container()
        
        prices_history = []
        demands_history = []
        revenues_history = []
        competitor_prices_history = []
        
        progress_bar = st.progress(0)
        
        while not done and step < simulation_steps:
            action = trainer.agent.select_action(state, training=False)
            next_state, reward, done, info = trainer.env.step(action)
            
            prices_history.append(info['price'])
            demands_history.append(info['demand'])
            revenues_history.append(info['revenue'])
            competitor_prices_history.append(info['competitor_price'])
            
            state = next_state
            step += 1
            progress_bar.progress(step / simulation_steps)
        
        st.session_state.simulation_running = False
        
        # Display results
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Price", f"${np.mean(prices_history):.2f}")
            with col2:
                st.metric("Avg Demand", f"{np.mean(demands_history):.1f}")
            with col3:
                st.metric("Total Revenue", f"${np.sum(revenues_history):,.0f}")
            with col4:
                st.metric("Avg Revenue/Step", f"${np.mean(revenues_history):.2f}")
        
        # Plot results
        with chart_container:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Price vs Time', 'Demand vs Time',
                              'Revenue vs Time', 'Price vs Demand'),
                vertical_spacing=0.12
            )
            
            steps = list(range(len(prices_history)))
            
            # Price
            fig.add_trace(
                go.Scatter(x=steps, y=prices_history, name='Our Price',
                          line=dict(color='#1f77b4')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=steps, y=competitor_prices_history, 
                          name='Competitor', line=dict(color='#ff7f0e', dash='dash')),
                row=1, col=1
            )
            
            # Demand
            fig.add_trace(
                go.Scatter(x=steps, y=demands_history, name='Demand',
                          line=dict(color='#2ca02c')),
                row=1, col=2
            )
            
            # Revenue
            fig.add_trace(
                go.Scatter(x=steps, y=revenues_history, name='Revenue',
                          line=dict(color='#d62728')),
                row=2, col=1
            )
            
            # Price-Demand scatter
            fig.add_trace(
                go.Scatter(x=prices_history, y=demands_history, mode='markers',
                          name='Price-Demand', marker=dict(color='#9467bd', size=5)),
                row=2, col=2
            )
            
            fig.update_xaxes(title_text="Time Step", row=2, col=1)
            fig.update_xaxes(title_text="Price ($)", row=2, col=2)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Demand", row=1, col=2)
            fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
            fig.update_yaxes(title_text="Demand", row=2, col=2)
            
            fig.update_layout(height=600, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)


def create_what_if_simulator():
    """Create what-if pricing simulator."""
    st.header("🔮 What-If Pricing Simulator")
    
    st.markdown("""
    Test different pricing strategies and see immediate demand and revenue predictions.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Parameters")
        
        manual_price = st.slider("Set Price ($)", 20.0, 100.0, 50.0, 1.0)
        time_of_day = st.slider("Hour of Day", 0, 23, 12)
        demand_level = st.slider("Current Demand Level", 0.0, 1.0, 0.5, 0.1)
        competitor_price = st.slider("Competitor Price ($)", 20.0, 100.0, 48.0, 1.0)
    
    with col2:
        st.subheader("Prediction Results")
        
        # Create environment for prediction
        env = MarketEnvironment(st.session_state.market_config)
        env.current_step = time_of_day
        env.previous_demand = demand_level * env.config.base_demand
        
        # Calculate predicted demand
        predicted_demand = env._calculate_demand(manual_price, competitor_price)
        predicted_revenue = manual_price * predicted_demand
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Demand", f"{predicted_demand:.1f} units")
        with col2:
            st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
        with col3:
            margin = ((manual_price - 30) / manual_price) * 100  # Assuming cost of $30
            st.metric("Profit Margin", f"{margin:.1f}%")
        
        # Sensitivity analysis
        st.subheader("Price Sensitivity Analysis")
        
        price_range = np.linspace(20, 100, 50)
        demands = [env._calculate_demand(p, competitor_price) for p in price_range]
        revenues = [p * d for p, d in zip(price_range, demands)]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Demand Curve', 'Revenue Curve')
        )
        
        # Demand curve
        fig.add_trace(
            go.Scatter(x=price_range, y=demands, mode='lines',
                      name='Demand', line=dict(color='#2ca02c', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[manual_price], y=[predicted_demand], mode='markers',
                      name='Current', marker=dict(color='red', size=12)),
            row=1, col=1
        )
        
        # Revenue curve
        fig.add_trace(
            go.Scatter(x=price_range, y=revenues, mode='lines',
                      name='Revenue', line=dict(color='#1f77b4', width=3)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[manual_price], y=[predicted_revenue], mode='markers',
                      name='Current', marker=dict(color='red', size=12)),
            row=1, col=2
        )
        
        # Find optimal price
        optimal_idx = np.argmax(revenues)
        optimal_price = price_range[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        fig.add_trace(
            go.Scatter(x=[optimal_price], y=[optimal_revenue], mode='markers',
                      name='Optimal', marker=dict(color='gold', size=12, 
                                                   symbol='star')),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Price ($)", row=1, col=1)
        fig.update_xaxes(title_text="Price ($)", row=1, col=2)
        fig.update_yaxes(title_text="Demand", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 **Optimal Price**: ${optimal_price:.2f} (Revenue: ${optimal_revenue:,.2f})")


def main():
    """Main application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">💰 RL Dynamic Pricing System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Reinforcement Learning for Smart Revenue Optimization</p>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    market_config = create_market_config_ui()
    st.session_state.market_config = market_config
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🎓 Training",
        "📈 Analysis",
        "🔴 Live Simulation",
        "🔮 What-If Analysis"
    ])
    
    with tab1:
        st.header("📊 System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Configuration")
            config_df = pd.DataFrame({
                'Parameter': ['Base Demand', 'Price Elasticity', 'Time Variance', 
                            'Competitor Influence', 'Noise StdDev'],
                'Value': [
                    f"{market_config.base_demand:.0f}",
                    f"{market_config.price_elasticity:.2f}",
                    f"{market_config.time_variance:.2f}",
                    f"{market_config.competitor_influence:.2f}",
                    f"{market_config.noise_std:.1f}"
                ]
            })
            st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Training Status")
            if st.session_state.trained_models:
                for model_name in st.session_state.trained_models.keys():
                    st.success(f"✅ {model_name.replace('_', ' ').title()} - Trained")
            else:
                st.info("ℹ️ No models trained yet. Go to the Training tab to get started!")
        
        if st.session_state.trained_models:
            st.subheader("Performance Summary")
            fig = plot_comparison(st.session_state.trained_models)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        create_training_ui()
        
        if st.session_state.trained_models:
            st.markdown("---")
            st.subheader("📈 Training Results")
            
            for model_name, data in st.session_state.trained_models.items():
                with st.expander(f"{model_name.replace('_', ' ').title()} Results", expanded=False):
                    fig = plot_training_progress(data['history'], 
                                                model_name.replace('_', ' ').title())
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Revenue", 
                                f"${data['evaluation']['mean_revenue']:,.0f}")
                    with col2:
                        st.metric("Std Revenue", 
                                f"${data['evaluation']['std_revenue']:,.0f}")
                    with col3:
                        st.metric("vs Baseline", 
                                f"+{data['comparison']['revenue_improvement_pct']:.1f}%")
    
    with tab3:
        st.header("📈 Detailed Analysis")
        
        if not st.session_state.trained_models:
            st.warning("⚠️ Please train models first to see analysis!")
        else:
            model_options = list(st.session_state.trained_models.keys())
            selected_model = st.selectbox(
                "Select Model for Analysis",
                [m.replace('_', ' ').title() for m in model_options],
                key='analysis_model'
            )
            
            model_key = selected_model.lower().replace(' ', '_')
            data = st.session_state.trained_models[model_key]
            
            # Exploration vs Exploitation
            st.subheader("🔍 Exploration vs Exploitation")
            
            epsilon_history = data['history']['epsilon_history']
            episodes = list(range(len(epsilon_history)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=episodes, y=epsilon_history,
                mode='lines', name='Epsilon',
                line=dict(color='#d62728', width=2),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title="Epsilon Decay Over Training",
                xaxis_title="Episode",
                yaxis_title="Epsilon (Exploration Rate)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Avg Reward", 
                        f"{np.mean(data['history']['episode_rewards'][-20:]):,.0f}")
            with col2:
                st.metric("Final Avg Revenue", 
                        f"${np.mean(data['history']['episode_revenues'][-20:]):,.0f}")
            with col3:
                st.metric("Final Avg Price", 
                        f"${np.mean(data['history']['episode_avg_prices'][-20:]):.2f}")
            with col4:
                st.metric("Baseline Improvement", 
                        f"+{data['comparison']['revenue_improvement_pct']:.1f}%")
    
    with tab4:
        create_live_simulation()
    
    with tab5:
        create_what_if_simulator()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>RL Dynamic Pricing System | Built with Streamlit & PyTorch</p>
        <p>💡 Tip: Adjust market parameters in the sidebar to see how they affect pricing strategies</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
