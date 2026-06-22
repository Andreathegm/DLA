import gymnasium as gym

class SmoothDrivingWrapper(gym.Wrapper):
    def __init__(self, env, max_consecutive_gas=15):
        super().__init__(env)
        self.last_action = None
        
        self.consecutive_gas_count = 0
        self.max_consecutive_gas = max_consecutive_gas

        self.ACTION_RIGHT = 1
        self.ACTION_LEFT = 2
        self.ACTION_GAS = 3

    def step(self, action):
        # Esegui lo step nel gioco (essendo in single-env, action è un singolo intero)
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        if self.last_action is not None:
            rapid_steering = (self.last_action == self.ACTION_LEFT and action == self.ACTION_RIGHT) or \
                             (self.last_action == self.ACTION_RIGHT and action == self.ACTION_LEFT)
            
            if rapid_steering:
                reward -= 0.5

        if action == self.ACTION_GAS:
            self.consecutive_gas_count += 1
        else:
            self.consecutive_gas_count = 0

        if self.consecutive_gas_count > self.max_consecutive_gas:
            reward -= 0.25

        self.last_action = action

        return next_obs, reward, terminated, truncated, info