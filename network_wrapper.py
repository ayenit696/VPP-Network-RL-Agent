import collections
import random
import gymnasium as gym

class MessyNetworkWrapper(gym.Wrapper):
    """
    This wrapper intercepts data between the real power grid and the AI.
    It introduces delayed observations (latency) and frozen data (packet drops).
    """
    def __init__(self, env, latency_steps=2, packet_drop_rate=0.15):
        # Initialize the parent gym.Wrapper class
        super().__init__(env)
        
        self.latency_steps = latency_steps
        self.packet_drop_rate = packet_drop_rate
        
        # A double-ended queue (deque) is perfect for latency. 
        # It acts as a pipe: new states go in the back, delayed states come out the front.
        self.buffer = collections.deque(maxlen=latency_steps + 1)
        self.last_delivered_obs = None

    def reset(self, **kwargs):
        # Reset the real physical power grid
        obs, info = self.env.reset(**kwargs)
        self.buffer.clear()
        
        # Prime our latency buffer with the initial starting state
        for _ in range(self.latency_steps + 1):
            self.buffer.append(obs)
            
        self.last_delivered_obs = obs
        return obs, info

    def step(self, action):
        # 1. Execute the AI's action in the real physical grid
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Add the brand new physical observation to the back of our latency queue
        self.buffer.append(next_obs)
        
        # 3. Pull the delayed observation from the front of our queue
        delayed_obs = self.buffer[0]
        
        # 4. Simulate a packet drop (unreliable cellular network)
        if random.random() < self.packet_drop_rate:
            # The packet dropped! The AI doesn't get the new data.
            # It must reuse the last successfully delivered observation (data freezing).
            info["packet_dropped"] = True
        else:
            # The packet arrived safely. Update our records.
            self.last_delivered_obs = delayed_obs
            info["packet_dropped"] = False
            
        # Return the stale/messy observation to the AI agent
        return self.last_delivered_obs, reward, terminated, truncated, info