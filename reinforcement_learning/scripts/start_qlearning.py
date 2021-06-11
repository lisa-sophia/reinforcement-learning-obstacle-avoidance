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
    rospy.loginfo("Starting Learning")

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
    Alpha = rospy.get_param("/turtlebot3/alpha")
    Epsilon = rospy.get_param("/turtlebot3/epsilon")
    Gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")
    min_alpha = rospy.get_param("/turtlebot3/min_alpha")
    max_alpha = rospy.get_param("/turtlebot3/max_alpha")

    running_step = rospy.get_param("/turtlebot3/running_step")

    complete_file_name = os.path.join(outdir, "rewards.txt")
    f = open(complete_file_name, "w+")
    f.write("")
    f.close()

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon
                           , use_q_table=True, file_dir=outdir)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
    max_steps_reached = False

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False
        qlearn.epsilon = Epsilon - epsilon_discount*(x/nepisodes)
        qlearn.alpha   = max_alpha - (max_alpha - min_alpha)*(x/nepisodes)
        #if qlearn.epsilon > 0.05:
        #    qlearn.epsilon *= epsilon_discount

        # If we start new episode because step limit was reached, set last episode to "done"
        if max_steps_reached:
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True
            max_steps_reached = False

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logdebug("Next action is:%d", action)
            rospy.logwarn("Picked action " + str(action) + " (0=forward, 1=left, 2=right, 3=stop)")
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
                # check if we're at the last possible step
                if i == (nsteps-1):
                    max_steps_reached = True
                    rospy.logerr("Reached max. number of steps ==> forced done")
            else:
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))

            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

        # save rewards in file, then we can plot that later
        complete_file_name = os.path.join(outdir, "rewards.txt")
        f = open(complete_file_name, "a+")
        f.write("%f\r\n" % (cumulated_reward))
        f.close()

        qlearn.saveQToFile(outdir, file_name="intermediate_q_table.pkl")

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.logwarn("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.logwarn("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    qlearn.saveQToFile(outdir)

    import subprocess
    cmd = '''killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient; for pid in $(ps -ef | grep "src/actor/src" | awk '{print $2}'); do kill -9 $pid; done'''
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')

    env.close()
