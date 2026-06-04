import streamlit as st
import gymnasium as gym
import gym_anm
import shimmy
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import time
from network_wrapper import MessyNetworkWrapper

# 1. Page Configuration
st.set_page_config(page_title="VPP AI Orchestration Dashboard", layout="wide")
st.title("⚡ Autonomous VPP Orchestration under Network Latency")
st.markdown("---")

# 2. Sidebar Configuration Controls
st.sidebar.header("📡 Network Degradation Controls")
latency_input = st.sidebar.slider("Network Latency (Steps)", min_value=0, max_value=5, value=2)
loss_input = st.sidebar.slider("Packet Drop Probability (%)", min_value=0, max_value=50, value=15) / 100.0

st.sidebar.header("🧠 Brain Optimization")
use_robust_model = st.sidebar.checkbox("Deploy Hardened Predictive AI", value=True)

# 3. Cache Environment and Models
@st.cache_resource
def init_simulation():
    gym.register_envs(shimmy)
    return gym.make("GymV21Environment-v0", env_id='ANM6Easy-v0')

@st.cache_resource
def load_brains():
    try:
        robust = PPO.load("vpp_anm_robust_model")
    except:
        robust = None
    return robust

base_env = init_simulation()
robust_ai = load_brains()

# 4. App Execution Flow
if st.button("▶️ Launch Real-Time Grid Orchestration"):
    if robust_ai is None:
        st.error("Missing model binary file! Ensure 'vpp_anm_robust_model.zip' is inside this directory.")
    else:
        # Initialize our messy wrapper natively using the user's live slider inputs
        env = MessyNetworkWrapper(base_env, latency_steps=latency_input, packet_drop_rate=loss_input)
        obs, info = env.reset()
        
        # Setup empty metric layout elements
        metric_row = st.columns(3)
        status_box = metric_row[0].empty()
        reward_box = metric_row[1].empty()
        loss_box = metric_row[2].empty()
        
        # Setup chart layout element
        chart_title = st.empty()
        chart_placeholder = st.empty()
        
        # Simulation loop state storage arrays
        reward_history = []
        network_status_history = []
        
        # 5. Live Simulation Run
        # for step in range(40):
        #     # Select model mode based on sidebar toggle
        #     if use_robust_model:
        #         action, _ = robust_ai.predict(obs, deterministic=True)
        #     else:
        #         action = env.action_space.sample() # Random action profile representing unhardened AI failing
                
        #     obs, reward, terminated, truncated, info = env.step(action)
            
        #     # Process dropped state tracking variables
        #     is_dropped = info.get("packet_dropped", False)
        #     reward_history.append(reward)
        #     network_status_history.append(1.0 if is_dropped else 0.0)
            
        #     # 6. Dynamic Streamlit Components UI Engine Redrawing Updates
        #     if is_dropped:
        #         status_box.metric("📶 Network Pipeline Status", "❌ PACKET LOST", delta="- Stale Observation Data Used", delta_color="inverse")
        #     else:
        #         status_box.metric("📶 Network Pipeline Status", "✅ DELIVERED", delta="Telemetry Synchronized")
                
        #     reward_box.metric("📊 Instant Operational Reward", f"{reward:.2f}", delta=f"Current Step: {step+1}")
        #     loss_box.metric("📉 Dynamic Target Jitter Boundary", f"{latency_input} Steps Lag")
            
        #     # Package updating analytics into data arrays
        #     chart_data = pd.DataFrame({
        #         "Grid Stability Reward": reward_history,
        #         "Packet Dropout Anomalies": network_status_history
        #     })
            
        #     chart_title.subheader("📈 Real-Time VPP Stability Diagnostics")
        #     chart_placeholder.line_chart(chart_data)
            
        #     # Artificial timing step lag to allow human eyes to track data flow transitions
        #     time.sleep(0.3)
            
        #     if terminated or truncated:
        #         st.error("🚨 CRITICAL BREAKDOWN: Physical grid voltage boundaries breached or transmission capacity overloaded.")
        #         break
        
        # if not (terminated or truncated):
        #     st.success("🏆 SUCCESSFUL EVALUATION RUN: The control node securely structured micro-grid profiles amidst communication degradation vectors.")
        # ... [Previous setup layout variables stay identical up to the loops section]

        # Simulation loop state historical arrays
        reward_history = []
        network_status_history = []
        
        step_counter = 0

        # Replace 'for step in range(40):' with an infinite while condition
        while True:
            step_counter += 1
            
            if use_robust_model:
                action, _ = robust_ai.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # Random failover mode
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            is_dropped = info.get("packet_dropped", False)
            reward_history.append(reward)
            network_status_history.append(1.0 if is_dropped else 0.0)
            
            # Keep history lengths manageable to protect browser memory performance
            if len(reward_history) > 100:
                reward_history.pop(0)
                network_status_history.pop(0)
            
            # Draw interactive dashboard layout rows
            if is_dropped:
                status_box.metric("📶 Network Pipeline Status", "❌ PACKET LOST", delta="- Stale Data Vector Injected", delta_color="inverse")
            else:
                status_box.metric("📶 Network Pipeline Status", "✅ DELIVERED", delta="Telemetry Streams Synced")
                
            reward_box.metric("📊 Instant Operational Reward", f"{reward:.2f}", delta=f"Total Steps: {step_counter}")
            loss_box.metric("降低 Network Lag Latency Boundary", f"{latency_input} Steps Lag")
            
            chart_data = pd.DataFrame({
                "Grid Stability Reward": reward_history,
                "Packet Dropout Anomalies": network_status_history
            })
            
            chart_title.subheader("📈 Endless Real-Time VPP Stability Diagnostics")
            chart_placeholder.line_chart(chart_data)
            
            import time
            time.sleep(0.15) # Adjust frame drawing refresh frequency speeds
            
            # Handle soft auto-reboot resets without breaking web render cycles
            if terminated or truncated:
                st.toast("🚨 Safety Boundary Tripped! Auto-rebooting grid topology.", icon="⚠️")
                obs, info = env.reset()