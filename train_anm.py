import gym_anm  # Secretly registers the old environment
import gymnasium as gym
import shimmy
from stable_baselines3 import PPO
from network_wrapper import MessyNetworkWrapper  # 1. Import our custom network layer

# 2. Initialize the Legacy-to-Modern Translation
gym.register_envs(shimmy)
base_env = gym.make("GymV21Environment-v0", env_id='ANM6Easy-v0')

# 3. Inject the Network Degradation Layer
# We pass the real physical power grid environment directly into our messy network wrapper
print("Injecting Network Degradation Layer (Latency & Packet Drops)...")
env = MessyNetworkWrapper(base_env, latency_steps=2, packet_drop_rate=0.15)

print("\nInitializing Advanced AI Model...")

# 4. Neural Network Policy Architecture
# 256x256 dense layers give the agent enough computational parameters to build 
# an internal 'Belief State' to forecast true state variables when packets drop.
policy_kwargs = dict(net_arch=[256, 256])

# Initialize the PPO agent with the upgraded brain on our wrapped environment
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# 5. Train the AI
print("Training AI to master grid physics while surviving an unstable network...")
# 100,000 timesteps allows the agent to experience thousands of random packet drops
model.learn(total_timesteps=100000)

# Save the final, optimized model
model.save("vpp_anm_robust_model")
print("\n[SUCCESS] Robust Advanced Physical Grid Model saved successfully!")