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
    env = gym.make('PandaReach-v2', control_type = "joint")  # ee (action space: 3) or joint (action space: 7)
    rospy.loginfo("MADE ENVIRONMENT")

    model_class = DDPG

    # The strategies for selecting new goals when creating artificial transitions. Available strategies: future, final, episode
    goal_selection_strategy = 'future'

    # If True the HER transitions will get sampled online.
    # data is available in sequential order.
    online_sampling = True
    # Time limit for the episodes
    max_episode_length = 1000

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
            max_episode_length = max_episode_length,
        ),
        verbose = 1,
        tensorboard_log="./reach_rl/"
    )

    # Train the model
    model.learn(15000)
    model.save("./reach_rl")