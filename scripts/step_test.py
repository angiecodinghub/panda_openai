#!/usr/bin/env python3

import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from stable_baselines3 import DDPG
#import my_reach_env
from openai_ros.task_envs.reach.reach import ReachEnv 


if __name__ == '__main__':

    rospy.init_node('step_test', anonymous = True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('PandaReach-v2')
    rospy.loginfo("Gym environment made")
    print("made env")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('panda_reach_test')
    outdir = pkg_path + '/testing_results'
    env = wrappers.Monitor(env, outdir, force = True)
    rospy.loginfo("Monitor Wrapper started")
    print("set log sys")

    # Load trained model
    model_class = DDPG
    model = model_class.load("/home/panda/catkin_ws/src/panda_reach_test/models/reach_vec", env = env)
    rospy.loginfo("Trained model loaded")
    print("loaded model")

    # start outputting path.
    obs = env.reset()
    #obs['observation'] = obs['observation'][:3]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic = True)#unexpected observation shape
        print("action:", action)
        obs, reward, done, _ = env.step(action)
        print("obs:", obs)
        #obs['observation'] = obs['observation'][:3]
        rospy.loginfo("observation:", obs)
        #env.render()

    #env.close()