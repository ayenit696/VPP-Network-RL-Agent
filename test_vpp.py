from vpp_env import MessyNetworkVPPEnv
from stable_baselines3 import PPO

# 1. Load the Environment
env = MessyNetworkVPPEnv(avg_latency=3, jitter_range=2, packet_loss_rate=0.15)

# 2. Load the Trained Brain
print("Loading trained AI...")
model = PPO.load("vpp_latency_model")

# 3. Test the AI in a Live Simulation
obs, info = env.reset()

print("\n--- Starting Grid Simulation ---")
# Let's test its real-time responses for 20 time steps
for step in range(20): 
    # deterministic=True tells the AI to use its best learned strategy, no more random guessing
    action, _states = model.predict(obs, deterministic=True)
    
    # The VPP agent applies the action to the grid
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Extract the data to read it easily
    grid_freq = obs[0]
    battery_charge = obs[1]
    
    # Print the AI's decision and the grid's physical reaction
    print(f"Step {step + 1}: Action = {action[0]:>5.2f} | Grid Freq = {grid_freq:>5.2f} | Battery = {battery_charge:>4.2f}")

    if terminated or truncated:
        print("Simulation ended early (Battery depleted or Overcharged).")
        break
        
print("--- Simulation Complete ---")