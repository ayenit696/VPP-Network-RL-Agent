import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MessyNetworkVPPEnv(gym.Env):
    def __init__(self, avg_latency=3, jitter_range=2, packet_loss_rate=0.15):
        super(MessyNetworkVPPEnv, self).__init__()
        
        # 1. Action & Observation Spaces (Same as before)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        
        # 2. Network Emulation Parameters
        self.avg_latency = avg_latency
        self.jitter_range = jitter_range
        self.packet_loss_rate = packet_loss_rate
        
        # A storage timeline of true states up to the maximum possible delay
        self.max_possible_delay = avg_latency + jitter_range + 1
        self.timeline = []
        
        # Track the last state the agent actually managed to receive
        self.last_received_state = None
        
        # Internal physical variables
        self.grid_frequency = 0.0
        self.battery_charge = 0.5
        self.current_step = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid_frequency = 0.0
        self.battery_charge = 0.5
        self.current_step = 0
        
        initial_state = np.array([self.grid_frequency, self.battery_charge], dtype=np.float32)
        self.last_received_state = initial_state
        
        # Initialize our timeline with the starting state
        self.timeline = [initial_state.copy() for _ in range(self.max_possible_delay)]
        
        return self.last_received_state, {}

    def step(self, action):
        self.current_step += 1
        
        # 3. Grid Physics
        grid_noise = np.random.normal(0, 0.1)
        battery_action = action[0]
        self.grid_frequency = grid_noise - (battery_action * 0.2)
        self.battery_charge = np.clip(self.battery_charge + (battery_action * 0.05), 0.0, 1.0)
        
        # Calculate Reward
        reward = -abs(self.grid_frequency)
        if self.battery_charge <= 0.01 or self.battery_charge >= 0.99:
            reward -= 1.0
            
        # Record the true current state into our history pipeline
        true_current_state = np.array([self.grid_frequency, self.battery_charge], dtype=np.float32)
        self.timeline.append(true_current_state.copy())
        if len(self.timeline) > self.max_possible_delay:
            self.timeline.pop(0)
            
        # 4. Network Emulation Logic (The "Messy" Channel)
        # Check if the packet is dropped entirely
        if np.random.rand() < self.packet_loss_rate:
            # Packet lost! The agent receives NO new data and relies on its old telemetry
            visible_state = self.last_received_state
        else:
            # Packet survived, but calculate a dynamic, fluctuating latency (Jitter)
            actual_delay = int(self.avg_latency + np.random.randint(-self.jitter_range, self.jitter_range + 1))
            actual_delay = max(0, min(actual_delay, self.max_possible_delay - 1))
            
            # Extract the delayed data point from our history timeline
            # An actual_delay of 0 means current state; 3 means 3 steps ago
            timeline_index = -(actual_delay + 1)
            visible_state = self.timeline[timeline_index]
            self.last_received_state = visible_state.copy()
            
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return visible_state, reward, terminated, truncated, {}