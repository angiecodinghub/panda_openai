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
    print("before init node")
    rospy.init_node('step_test', anonymous = True, log_level=rospy.DEBUG)
    print("after init node")
    # Create the Gym environment
    env = gym.make('PandaReach-v2')
    rospy.loginfo("Gym environment made")

    # Load trained model
    model_class = DDPG
    model = model_class.load("/home/panda/catkin_ws/src/panda_reach_test/models/reach_vec", env = env)
    rospy.loginfo("Trained model loaded")

    # start outputting path.
    obs = env.reset() # change this to init position?
    done = False
    action = np.array([0.0032298252917826176, -0.14821836352348328, 0.9403465390205383])
    obs, reward, done, _ = env.step(action)
    #print(obs)
    # Failed to validate trajectory: couldn't receive full current joint state within 1s
    # Failed to fetch current robot state.
    """
    while not done:
        # action, _ = model.predict(obs, deterministic = True) # unexpected observation shape
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print("step test new observation:\n", obs['observation'])    
    """
