from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from my_hexabullet_env import HexapodBulletEnv
import math
import pybullet as p
import matplotlib.pyplot as plt

class RewardLoggerCallback(BaseCallback):
    """
    Custom callback to log episode rewards.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

def train(env, exp_name, **kwargs):
    all_rewards = []  # Store rewards from the run

    # Run only once
    log_dir = "tmp/"
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=32,
        tensorboard_log=log_dir,
        **kwargs
    )
    print("[+] Starting training")

    # Create callback for logging rewards
    reward_callback = RewardLoggerCallback()

    model.learn(
        total_timesteps=100000,
        log_interval=1,
        tb_log_name="PPO",
        callback=reward_callback
    )

    # Store rewards from this run
    all_rewards.append(reward_callback.episode_rewards)

    model.save(f"trained_models/{exp_name}_0")
    
    env.close()
    del env
    env = HexapodBulletEnv(
        client,
        time_step=0.25,
        frameskip=12,
        render=False,
        max_velocity=59 * 2 * math.pi / 60,
        max_torque=1.50041745
    )
    del model

    # After training, plot rewards
    plt.plot(all_rewards[0], label="Run 0")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    client = p.connect(p.DIRECT)
    env = HexapodBulletEnv(
        client,
        time_step=0.25,
        frameskip=12,
        render=False,
        max_velocity=59 * 2 * math.pi / 60,
        max_torque=1.50041745
    )
    train(
        env=env,
        exp_name="test",
        gamma=0.99,
        n_steps=128,
        batch_size=32,
        ent_coef=0.01,
        learning_rate=1e-3,
        clip_range=0.2,
        device="auto",
        _init_setup_model=True
    )