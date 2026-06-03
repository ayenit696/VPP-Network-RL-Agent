import gym_anm  
import gymnasium as gym
import shimmy
from stable_baselines3 import PPO
from network_wrapper import MessyNetworkWrapper

# 1. Initialize the physical grid and the messy network layer
gym.register_envs(shimmy)
base_env = gym.make("GymV21Environment-v0", env_id='ANM6Easy-v0')

# We must test the AI in the exact same messy conditions it trained in!
env = MessyNetworkWrapper(base_env, latency_steps=2, packet_drop_rate=0.15)

# 2. Load the Hardened Brain
print("Loading the Robust VPP Engine...")
model = PPO.load("vpp_anm_robust_model")

# 3. Start the Live Simulation
obs, info = env.reset()
print("\n" + "="*50)
print(" ⚡ INITIATING LIVE GRID ORCHESTRATION ⚡")
print("="*50 + "\n")

# We will let the AI run the grid for 30 time steps
for step in range(30):
    # deterministic=True: The AI uses its absolute best strategy, no random guessing
    action, _states = model.predict(obs, deterministic=True)
    
    # The AI sends its command through the messy network to the physical grid
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check if our network wrapper dropped the telemetry packet this step
    packet_status = "❌ DATA LOST (Using Stale State)" if info.get("packet_dropped") else "✅ Telemetry Delivered"
    
    print(f"Step {step + 1:02d} | Network: {packet_status:<32} | Grid Status Reward: {reward:>.2f}")
    
    # If the AI accidentally breaches thermal limits, the grid crashes (terminated=True)
    if terminated or truncated:
        print("\n🚨 CRITICAL FAILURE: Physical Grid Limits Breached. Blackout triggered.")
        break

print("\n--- Simulation Complete ---")
if not terminated:
    print("🏆 SUCCESS: The AI successfully maintained grid stability without melting the power lines!")