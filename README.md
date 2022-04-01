# panda_openai

## Capabilities
 
This **ROS Noetic** package is where we put all the training scripts in.

## Installation

Execute the following commands:<br>
`cd ~/ros_ws/src`<br>
`git clone https://github.com/angiecodinghub/panda_openai.git`<br>
`cd ~/ros_ws`<br>
`catkin_make`<br>
`source devel/setup.bash`<br>
`rosdep install panda_openai`<br>

The following packages will need to be installed as well to get the full functionality of what we want:<br>
1. [openai_ros](https://github.com/angiecodinghub/openai_ros)
2. [panda_moveit_config](https://github.com/angiecodinghub/panda_moveit_config)
3. [franka_ros](https://github.com/frankaemika/franka_ros)
4. moveit

The following python3 modules will need to be installed as well:<br>
1. rospy
2. gym == 0.20.0
3. stable_baselines3

## Example Usage
1. load model and perform step function:
```python
    rospy.init_node('step_test', anonymous = True)

    env = gym.make('PandaReach-v2', control_type = "joint")

    model_class = DDPG
    model = model_class.load("$(path to your model)", env = env)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic = True)
        obs, reward, done, _ = env.step(action)

    env.close()
```
2. train a reinforcement model:
```python
    rospy.init_node('reach_rl', anonymous = True)

    env = gym.make('PandaReach-v2', control_type = "joint")

    model_class = DDPG
    goal_selection_strategy = 'future'
    online_sampling = True
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class = HerReplayBuffer,

        replay_buffer_kwargs = dict(
            n_sampled_goal = 4,
            goal_selection_strategy = goal_selection_strategy,
            online_sampling = online_sampling,
        ),
        verbose = 1,
    )

    model.learn(1000)
```
3. how to launch a node:<br>
You can get the full functionality as seen in the demo down below by running these 3 commands in order:
* roslaunch panda_gazebo panda.launch (headless:=True)
* roslaunch panda_moveit_config panda_moveit.launch (load_gripper:=False)
* roslaunch panda_openai ($ name of your launch file)
## Demo
1. step function:<br>
https://user-images.githubusercontent.com/61912547/160753940-38d11452-d68e-4303-892c-de86cde610a1.mp4

2. training a model:<br>
https://user-images.githubusercontent.com/61912547/160755796-f5470549-598f-478c-a813-d7f525dbe37d.mp4

## Unresolved Bugs

1. [HALT] The step function takes quite long to execute. It's in the 0.1 scale while that of the panda-gym package is in the 0.001 scale. The following command is the bottleneck (in openai_ros):
```python
result = self.group.go(wait = True)
```
This is cause by the ```wait = True``` flag; when it is set as ```false```, it takes 0f 0.001 scale to execute. However, it's necessary to set ```wait = True``` here to get the correct observation space, so we'll leave it as it is for now.

2. [ONGOING] The same issue mentioned [here](https://answers.ros.org/question/273871/controller-aborts-trajectory-goal-with-goal_tolerance_violation-after-execution/). The performance doesn't change, but the warning messages are super annoying.

3. [ONGOING] For the "ee" action space, the planned path from MoveIt are lengthy and unnecessary. A demo video can be seen here:<br>
https://user-images.githubusercontent.com/61912547/160915547-5f70eae9-ac67-4de0-ab9f-f889f945c626.mp4

4. [ONGIONG] When training, error "is the target within bounds?" appear when the robot is quite stretched, or enccountered a collision. However, the action space has been clipped to keep within bound. Two example figures are attached:

## Contact Info

Maintainer: Angela Wu (annwu@rice.edu) <br>
Reachable via Slack as well.
