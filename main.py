import os
import supersuit as ss
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v6
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

# Define model save path and timesteps per checkpoint
MODEL_PATH = "ppo_pistonball"
CHECKPOINT_INTERVAL = 50000  # Save the model every 50,000 timesteps
TOTAL_TIMESTEPS = 200000     # Total timesteps for training

# Initialize the environment
env = pistonball_v6.parallel_env()
env = ss.color_reduction_v0(env)
env = ss.dtype_v0(env, 'float32')  # Ensure observation space is float32
env = ss.normalize_obs_v0(env)
env = ss.frame_stack_v1(env, 3)

# Create a DummyVecEnv for vectorization
num_envs = 4  # Adjust this to the number of parallel environments you want
env = DummyVecEnv([lambda: env for _ in range(num_envs)])  # Wrap the environment for vectorization

# Add VecMonitor for logging
env = VecMonitor(env)  # Add monitor wrapper for logging

# Check the environment
check_env(env, warn=True)

# Check if a model already exists
if os.path.exists(MODEL_PATH + ".zip"):
    print(f"Loading model from {MODEL_PATH}.zip")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("No existing model found, creating a new one.")
    model = PPO('MlpPolicy', env, verbose=1)

# Continue training with saving checkpoints
timesteps_trained = 0
while timesteps_trained < TOTAL_TIMESTEPS:
    model.learn(total_timesteps=CHECKPOINT_INTERVAL)
    timesteps_trained += CHECKPOINT_INTERVAL
    print(f"Checkpoint: saving model at {timesteps_trained} timesteps.")
    model.save(MODEL_PATH)

print(f"Training complete, model saved as {MODEL_PATH}.zip")
