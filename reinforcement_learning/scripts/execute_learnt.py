#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from functools import reduce
import os

if __name__ == '__main__':

    rospy.init_node('turtlebot3_pedestrians_qlearn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('reinforcement_learning')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Epsilon = 0.0
    Alpha   = rospy.get_param("/turtlebot3/min_alpha")
    Gamma   = rospy.get_param("/turtlebot3/gamma")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes_execution")
    nsteps    = rospy.get_param("/turtlebot3/nsteps")
    
    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon,
                           use_q_table=True, file_dir=outdir)

    start_time = time.time()
    highest_reward = 0

    # store the rewards to see how often we succeeded
    complete_file_name = os.path.join(outdir, "rewards_execution.txt")
    f = open(complete_file_name, "w+")
    f.write("")
    f.close()

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        start_time_episode = time.time()
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        actions = []

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logdebug("Next action is:%d", action)
            actions.append(action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logdebug("NOT DONE")
                state = nextState
            else:
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        end_time_episode = time.time()
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

        # save rewards in file, then we can plot that later
        complete_file_name = os.path.join(outdir, "rewards_execution.txt")
        f = open(complete_file_name, "a+")
        f.write("%f\r\n" % (cumulated_reward))
        f.close()

        # save log output in file
        complete_file_name = os.path.join(outdir, "log_output_execution.txt")
        f = open(complete_file_name, "a+")
        f.write( ("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
                round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
                cumulated_reward) + " - Actions:" + str(actions) + " - Duration:" + str(end_time_episode-start_time_episode)
                + "     Time: %d:%02d:%02d" % (h, m, s) + "\n") )
        f.close()

    l = last_time_steps.tolist()
    l.sort()

    #qlearn.saveQToFile(outdir)

    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))

    import subprocess
    cmd = '''killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient; for pid in $(ps -ef | grep "src/actor/src" | awk '{print $2}'); do kill -9 $pid; done'''
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
    
    env.close()