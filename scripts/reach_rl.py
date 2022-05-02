#!/usr/bin/env python3

import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
#import my_reach_env
from openai_ros.task_envs.reach.reach import ReachEnv 
import time
import numpy as np

if __name__ == '__main__':
    # init ROS node.
    rospy.init_node('reach_rl', anonymous = True, log_level=rospy.INFO)
    
    # Create the Gym environment
    env = gym.make('PandaReach-v2', control_type = "joint", robot_type = "real")  # ee (action space: 3) or joint (action space: 7)
    rospy.loginfo("MADE ENVIRONMENT")

    model_class = DDPG

    # The strategies for selecting new goals when creating artificial transitions. Available strategies: future, final, episode
    goal_selection_strategy = 'future'

    # If True the HER transitions will get sampled online.
    # data is available in sequential order.
    online_sampling = True

    # Initialize the model
    model = model_class(
        "MultiInputPolicy", # Dict Observation space. Has multiple features.
        env,
        replay_buffer_class = HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs = dict(
            n_sampled_goal = 4, # # artificial transitions to generate for each actual transition.
            goal_selection_strategy = goal_selection_strategy,
            online_sampling = online_sampling,
        ),
        verbose = 1,
        # tensorboard_log="./reach_rl/"
    )
    rospy.loginfo("MODEL INITIALIZED")

    # Train the model
    model.learn(10000) # 100: timesteps.
# model.save("/home/panda/catkin_ws/src/panda_openai/models/reach_rl_10000")
# reach_rl_10000's last stats:
# ---------------------------------
# | rollout/           |          |
# |    ep_len_mean     | 2        |
# |    ep_rew_mean     | -1       |
# | time/              |          |
# |    episodes        | 4528     |
# |    fps             | 0        |
# |    time_elapsed    | 16285    |
# |    total_timesteps | 9997     |
# | train/             |          |
# |    actor_loss      | 0.123    |
# |    critic_loss     | 0.00758  |
# |    learning_rate   | 0.001    |
# |    n_updates       | 9901     |
# ---------------------------------
