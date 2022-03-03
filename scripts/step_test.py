#!/usr/bin/env python

import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.reach import ReachEnv
from stable_baselines3 import DDPG


if __name__ == '__main__':

    rospy.init_node('step_test', anonymous = True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('PandaReach-v2')
    rospy.loginfo("Gym environment made")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('panda_reach_test')
    outdir = pkg_path + '/testing_results'
    env = wrappers.Monitor(env, outdir, force = True)
    rospy.loginfo("Monitor Wrapper started")

    # Load trained model
    model_class = DDPG
    model = model_class.load("./models/reach_vec", env = env)
    rospy.loginfo("Trained model loaded")
    
    # start outputting path.
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic = True)
        obs, reward, done, _ = env.step(action)
        rospy.loginfo("observation:", obs)
        env.render()

    env.close()