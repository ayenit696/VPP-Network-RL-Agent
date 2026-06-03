from vpp_env import MessyNetworkVPPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Initialize the upgraded environment
# 15% packet loss and up to 5 steps of sudden jitter spikes!
env = MessyNetworkVPPEnv(avg_latency=3, jitter_range=2, packet_loss_rate=0.15)

print("Initializing Robust AI Engine...")
# We use n_steps=2048 to allow PPO to look at a wider window of experiences before updating,
# which helps it ignore random network anomalies.
model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, verbose=1)

print("Training AI to handle unpredictable packet loss and jitter...")
model.learn(total_timesteps=60000)

print("Evaluating Robust AI...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Average Robust Score: {mean_reward} +/- {std_reward}")

model.save("vpp_robust_model")
print("Robust model saved successfully!")