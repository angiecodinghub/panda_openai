#!/usr/bin/env python3

import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
#import my_reach_env
from openai_ros.task_envs.reach.reach import ReachEnv 
import time
import numpy as np

if __name__ == '__main__':
    # init ROS node.
    rospy.init_node('step_test', anonymous = True, log_level=rospy.INFO)

    # Create the Gym environment
    env = gym.make('PandaReach-v2', control_type = "joint")  # ee (action space: 3) or joint (action space: 7)
    rospy.loginfo("MADE ENVIRONMENT")

    # Load trained model
    model_class = DDPG
    model = model_class.load("/home/panda/catkin_ws/src/panda_reach_test/models/reach_joint", env = env)
    rospy.loginfo("LOADED MODEL")

    # start outputting path.
    obs = env.reset() # init position will be the one with 90 degree at the elbow.
    done = False
    rospy.loginfo("GOAL OBSERVATION:")
    rospy.loginfo(obs["desired_goal"])
    rospy.loginfo("START OBSERVATION:")
    rospy.loginfo(obs['observation'])
    while not done:
        action, _ = model.predict(obs, deterministic = True)
        obs, reward, done, _ = env.step(action)
        rospy.loginfo("ACTION")
        rospy.loginfo(action)
        rospy.loginfo("NEW OBSERVATION:")
        rospy.loginfo(obs['observation'])

    ##############    
    # reachable action for "ee" type
    # action = np.array([0.0032298252917826176, -0.14821836352348328, 0.9403465390205383])
    # action = np.array([0.001, 0.001, 0.001])
    ##############  