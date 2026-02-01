# 🚀 Quick Start Guide

Get the RL Dynamic Pricing System running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 500MB free disk space

## Installation

### Option 1: Local Installation (Recommended for Development)

```bash
# 1. Clone or download the repository
cd rl_pricing_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Option 2: Docker (Recommended for Production)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

### Option 3: Streamlit Cloud (Recommended for Sharing)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your repository
4. Share the URL!

## First Steps

### 1. Configure Market (2 minutes)

Open the sidebar and adjust:
- **Base Demand**: 100 (average customers)
- **Price Elasticity**: -1.5 (demand sensitivity)
- **Time Variance**: 0.3 (demand fluctuation)

### 2. Train Your First Model (3 minutes)

1. Go to **Training** tab
2. Select **"Deep Q-Network (DQN)"**
3. Set **Training Episodes**: 100
4. Click **"Start Training"**

Wait ~1 minute for training to complete.

### 3. View Results (1 minute)

Navigate through tabs:
- **Dashboard**: See performance comparison
- **Analysis**: View learning curves
- **Live Simulation**: Watch agent make pricing decisions
- **What-If**: Test custom scenarios

## Understanding the Dashboard

### Dashboard Tab
Shows overall system status and model performance comparison.

**Key Metrics**:
- ✅ Trained models status
- 📊 Revenue comparison chart
- 📈 Improvement vs baseline

### Training Tab
Train and compare RL algorithms.

**What to do**:
1. Select algorithm (Q-Learning or DQN)
2. Set training episodes (50-200 recommended)
3. Click "Start Training"
4. View training curves

**Expected Results**:
- DQN typically achieves 15-25% revenue improvement
- Q-Learning achieves 10-20% improvement
- Training takes 30s-2min depending on episodes

### Analysis Tab
Detailed performance metrics and learning progress.

**What to look for**:
- **Epsilon decay**: Should decrease smoothly
- **Episode rewards**: Should increase over time
- **Revenue**: Should stabilize at higher levels

### Live Simulation Tab
Real-time pricing simulation.

**How to use**:
1. Select trained model
2. Set simulation steps (100-500)
3. Click "Run Simulation"
4. Watch real-time pricing decisions

**What you'll see**:
- Price adjustments over time
- Demand response
- Revenue accumulation
- Competitor interactions

### What-If Simulator
Test pricing scenarios instantly.

**Try this**:
1. Set price to $45
2. Adjust hour of day
3. Change competitor price
4. See predicted demand & revenue

**Experiment with**:
- Different prices
- Peak vs off-peak hours
- Competitive positioning

## Common Scenarios

### Scenario 1: E-commerce Product Pricing

**Configuration**:
```
Base Demand: 150
Price Elasticity: -2.0 (highly elastic)
Time Variance: 0.5 (high daily variation)
Competitor Influence: 0.6 (competitive market)
```

**Expected Behavior**:
- Prices lower during low-demand periods
- Prices higher during peak hours
- Quick response to competitor changes

### Scenario 2: Hotel Room Pricing

**Configuration**:
```
Base Demand: 80
Price Elasticity: -1.2 (moderately elastic)
Time Variance: 0.7 (weekly patterns)
Competitor Influence: 0.3 (brand loyalty)
```

**Expected Behavior**:
- Higher prices on weekends
- Stable prices during weekdays
- Less reactive to competitors

### Scenario 3: Luxury Product Pricing

**Configuration**:
```
Base Demand: 50
Price Elasticity: -0.8 (inelastic)
Time Variance: 0.2 (stable demand)
Competitor Influence: 0.2 (brand strength)
```

**Expected Behavior**:
- Higher stable prices
- Minimal price changes
- Brand positioning maintained

## Tips for Best Results

### Training Tips

1. **Start Small**: Begin with 50-100 episodes
2. **Compare Algorithms**: Train both Q-Learning and DQN
3. **Watch Convergence**: Training should show improvement
4. **Patience**: Let epsilon decay for better exploration

### Configuration Tips

1. **Realistic Elasticity**: Use -1.5 to -2.0 for most products
2. **Time Variance**: 0.3-0.5 captures daily patterns
3. **Competitor Influence**: 0.3-0.6 for competitive markets

### Analysis Tips

1. **Check Improvement**: Should see 10-25% revenue gain
2. **Monitor Epsilon**: Should decay to 0.01-0.1
3. **Examine Prices**: Should make business sense
4. **Test What-If**: Validate agent's decisions

## Troubleshooting

### Issue: Training is slow

**Solution**:
- Reduce number of episodes
- Use Q-Learning instead of DQN
- Close other applications

### Issue: No improvement over baseline

**Solutions**:
- Train longer (200+ episodes)
- Adjust learning rate (try 0.05 or 0.2)
- Check market configuration (realistic elasticity)

### Issue: Prices seem random

**Solutions**:
- Train longer for convergence
- Increase epsilon_decay (more exploitation)
- Use lower learning rate

### Issue: Application won't start

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+

# Run with verbose logging
streamlit run app.py --logger.level=debug
```

## Next Steps

### Beginner
1. ✅ Run first training
2. ✅ Compare both algorithms
3. ✅ Try different market configs
4. ✅ Understand visualizations

### Intermediate
1. Train for 200+ episodes
2. Compare multiple market scenarios
3. Analyze learning curves
4. Test edge cases in What-If

### Advanced
1. Modify market environment
2. Create custom reward functions
3. Implement new algorithms
4. Deploy to production

## Learning Resources

### Understanding RL Concepts

**Q-Learning**:
- Learns value of state-action pairs
- Simple and interpretable
- Works well for discrete spaces

**DQN**:
- Uses neural networks
- Better generalization
- Scales to complex problems

**Epsilon-Greedy**:
- Balances exploration vs exploitation
- High epsilon = more exploration
- Decays over time to converge

### Key Metrics Explained

**Episode Reward**:
- Sum of all rewards in episode
- Should increase during training
- Includes revenue and penalties

**Revenue**:
- Price × Demand per step
- Primary business metric
- Compared against baseline

**Improvement %**:
- RL revenue vs baseline revenue
- 10-25% is typical
- Higher in volatile markets

## FAQs

**Q: How long should I train?**
A: 100-200 episodes is usually sufficient. Watch for convergence.

**Q: Which algorithm is better?**
A: DQN typically performs better but takes longer. Start with both.

**Q: Can I use real data?**
A: Yes! Modify the environment to use your demand model.

**Q: How do I deploy this?**
A: See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Q: Can I add more features?**
A: Absolutely! The code is modular and extensible.

**Q: Is this production-ready?**
A: The code is production-quality, but test thoroughly with your data.

## Performance Expectations

### Typical Results

**After 100 episodes**:
- DQN: 12-18% improvement
- Q-Learning: 8-15% improvement

**After 200 episodes**:
- DQN: 18-25% improvement
- Q-Learning: 12-20% improvement

**Training Time**:
- Q-Learning: ~30 seconds for 200 episodes
- DQN: ~2 minutes for 200 episodes

**Convergence**:
- Usually within 100-150 episodes
- Check epsilon decay curve
- Rewards should stabilize

## Getting Help

### Documentation
- [README.md](README.md) - Full documentation
- [API.md](API.md) - Technical reference
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design

### Support
- GitHub Issues: Report bugs
- Discussions: Ask questions
- Email: your-email@example.com

### Community
- Share your results
- Contribute improvements
- Help other users

## Keyboard Shortcuts

- `Ctrl+R` / `Cmd+R`: Refresh dashboard
- `Ctrl+S` / `Cmd+S`: Stop running script
- `C`: Clear cache (in Streamlit menu)

## Best Practices

1. **Start Simple**: Use default settings first
2. **Iterate**: Make small configuration changes
3. **Validate**: Check results make business sense
4. **Document**: Save successful configurations
5. **Monitor**: Track performance over time

---

**You're Ready!** 🎉

Start exploring the system and watch your RL agents learn optimal pricing strategies!

Need help? Check the [full documentation](README.md) or open an issue.

Happy pricing! 💰
