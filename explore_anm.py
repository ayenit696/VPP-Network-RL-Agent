import gym_anm  # This secretly registers the grid in the old 'gym' background
import gymnasium as gym
import shimmy

# Tell modern Gymnasium to use the Shimmy translation layer
gym.register_envs(shimmy)

# 1. Initialize the legacy simulation using the V21 translator wrapper
print("Bridging legacy Gym with modern Gymnasium...")
env = gym.make("GymV21Environment-v0", env_id='ANM6Easy-v0')
obs, info = env.reset()

print("--- Real Power Grid Physics Initialized ---")

# 2. Look at the Observation Space
print(f"Number of sensor data points the AI must now process: {env.observation_space.shape[0]}")

# 3. Look at the Action Space
print(f"Number of distinct control levers the AI must manage: {env.action_space.shape[0]}")

print("\nSample Data Array from the Grid:")
print(obs)