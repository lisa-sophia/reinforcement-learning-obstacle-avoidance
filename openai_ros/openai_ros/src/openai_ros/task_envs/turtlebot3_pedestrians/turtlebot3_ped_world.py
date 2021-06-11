import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot3_pedestrians_env
from gym.envs.registration import register
from geometry_msgs.msg import Vector3
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from nav_msgs.msg import Odometry
import os
import tf
import geometry_msgs
import math

class TurtleBot3PedestriansEnv(turtlebot3_pedestrians_env.TurtleBot3PedEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot3/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="simulation",
                    launch_file_name="minimal_sim.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/turtlebot3_pedestrians/config",
                               yaml_file_name="turtlebot3_world.yaml")


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TurtleBot3PedestriansEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot3/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)


        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot3/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot3/init_linear_turn_speed')

        self.new_ranges = rospy.get_param('/turtlebot3/new_ranges')
        self.min_range = rospy.get_param('/turtlebot3/min_range')
        self.max_laser_value = rospy.get_param('/turtlebot3/max_laser_value')
        self.min_laser_value = rospy.get_param('/turtlebot3/min_laser_value')
        self.max_linear_aceleration = rospy.get_param('/turtlebot3/max_linear_aceleration')

        # Get Desired Point to Get
        self.desired_point = geometry_msgs.msg.Point()
        self.desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")

        self.step_size = rospy.get_param("/turtlebot3/step_size")
        self.comfortable_distance = rospy.get_param("/turtlebot3/pedestrian_comfort_zone")

        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        #laser_scan = self.get_laser_scan()
        #num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        #high = numpy.full((num_laser_readings), self.max_laser_value)
        #low = numpy.full((num_laser_readings), self.min_laser_value)

        # We only use two integers
        #self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        #rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards
        self.forwards_reward = rospy.get_param("/turtlebot3/forwards_reward")
        self.turn_reward = rospy.get_param("/turtlebot3/turn_reward")
        self.end_episode_points = rospy.get_param("/turtlebot3/end_episode_points")

        self.cumulated_steps = 0.0


    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        odometry = self.get_valid_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position)


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot3
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_RIGHT"
        elif action == 3: #STOP
            linear_speed = 0.0
            angular_speed = 0.0
            self.last_action = "STOP"

        # We tell TurtleBot3 the linear and angular speed to set to execute
        #self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        start_pose = self.get_valid_odom()
        self.custom_move_base(start_pose, self.last_action, self.linear_forward_speed, self.linear_turn_speed, self.angular_speed, self.step_size)

        rospy.logdebug("END Set Action ==>"+str(action))


    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot3Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        self._episode_done = self.has_crashed_scan(laser_scan)
        #discretized_laser_obs = self.discretize_scan_observation(laser_scan, self.new_ranges)

        # Get odometry (current position) of the robot
        odometry = self.get_valid_odom(max_retries=100)
        if odometry is None:
            return None
        # We only care about the current x and y position of the robot
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        discretized_yaw = self.discretize_yaw(odometry.pose.pose.orientation)

        # We round to only one decimal to avoid very big Observation space
        #odometry_array = [round(x_position, 1), round(y_position, 1), discretized_yaw]
        odometry_array = [self.round_to_base(x_position, 0.5), self.round_to_base(y_position, 0.5), discretized_yaw]

        # Then also get state (x,y,yaw) and speed of the closest pedestrian
        ped_state = self.get_closest_pedestrian_state()
        ped_x = ped_state.position.x
        ped_y = ped_state.position.y
        quaternion = (ped_state.orientation.x,
                      ped_state.orientation.y,
                      ped_state.orientation.z,
                      ped_state.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        ped_yaw = euler[2]
        ped_speed = self.get_closest_pedestrian_speed()
        if ped_speed is None:
            rospy.logwarn("no pedestrian within 3 metre radius found")

        # Again, round to avoid very big Observation space
        # pedestrian might have value None, if there is no pedestrian between goal and robot
        # -> if it is none, we fill every value with '99', to showing that pedestrian is infinitely far away
        if ped_x is None or ped_y is None or ped_speed is None or ped_yaw is None:
            ped_x     = int(99)
            ped_y     = int(99)
            ped_speed = int(99)
            ped_yaw   = int(99)
        else:
            # save relative pedestrian coordinates
            ped_x     = ped_x - x_position
            #ped_x     = round(ped_x, 1)
            ped_x     = self.round_to_base(ped_x, 0.5)
            ped_y     = ped_y - y_position
            #ped_y     = round(ped_y, 1)
            ped_y     = self.round_to_base(ped_y, 0.5)
            ped_speed = round(ped_speed, 3)
            ped_yaw   = int(ped_yaw)

        pedestrian_array = [ped_x, ped_y, ped_yaw, ped_speed]

        #observations = discretized_laser_obs + odometry_array + pedestrian_array
        # we remove the laser scan data, since the x-y position should be enough (the robot will then learn where the walls are)
        observations = odometry_array + pedestrian_array
        rospy.logwarn("CURRENT STATE = " + str(observations))
        rospy.logdebug("Observations==>"+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    
    def _is_done(self, observations):

        if self._episode_done:
            rospy.logerr("TurtleBot3 is too close to wall==>")
        else:
            rospy.logwarn("TurtleBot3 is NOT close to a wall ==>")

        # Now we check if it has crashed based on the imu
        #imu_data = self.get_imu()
        #linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        #if linear_acceleration_magnitude > self.max_linear_aceleration:
        #    rospy.logerr("TurtleBot3 Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
        #    self._episode_done = True
        #else:
        #    rospy.logerr("DIDNT crash TurtleBot3 ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))

        current_position = geometry_msgs.msg.Point()
        current_position.x = observations[-7]
        current_position.y = observations[-6]
        current_position.z = 0.0

        # The learning space, we should not leave the corridor
        MAX_X = self.desired_point.x + 0.5
        MIN_X = -0.5
        MAX_Y = 1.5
        MIN_Y = -1.5

        # We see if we are outside the Learning Space
        if current_position.x <= MAX_X and current_position.x >= MIN_X:
            if current_position.y <= MAX_Y and current_position.y >= MIN_Y:
                rospy.logdebug("TurtleBot Position is OK ==>["+str(current_position.x)+","+str(current_position.y)+"]")
                
                # We see if it got to the desired point
                if self.is_in_desired_position(current_position, epsilon=self.step_size/2):
                    self._episode_done = True

                # We see if we have crashed with the closest pedestrian
                distance_from_pedestrian = self.get_distance_from_pedestrian(current_position, observations[-4:])
                rospy.logwarn("DISTANCE TO CLOSEST PEDESTRIAN = " + str(distance_from_pedestrian))
                if (distance_from_pedestrian < self.step_size): #0.25):
                    rospy.logerr("TurtleBot has crashed with pedestrian ==> distance was " + str(distance_from_pedestrian))
                    self._episode_done = True
            
            else:
                rospy.logerr("TurtleBot too far in Y Pos ==>"+str(current_position.y))
                self._episode_done = True
        else:
            rospy.logerr("TurtleBot too far in X Pos ==>"+str(current_position.x))
            self._episode_done = True

        return self._episode_done


    def _compute_reward(self, observations, done):

        reward = 0

        current_position = geometry_msgs.msg.Point()
        current_position.x = observations[-7]
        current_position.y = observations[-6]
        current_position.z = 0.0

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point

        # shortest separation distance between the robot and humans
        distance_from_pedestrian = abs(self.get_distance_from_pedestrian(current_position, observations[-4:]))

        if not done:
            if distance_from_pedestrian < self.comfortable_distance:
                reward = 0.05 * (distance_from_pedestrian - self.comfortable_distance) # was 0.5 * before
            # we give a small negative reward at each step, to make it more desirable to reach the goal in fewest steps possible
            # minus 1% of reward for reaching the goal
            reward -= 0.01*self.end_episode_points

            # If there has been a decrease in the distance to the desired point, we reward it
            #if distance_difference < 0.0:
            #    rospy.logwarn("DECREASE IN DISTANCE GOOD! REDUCED DISTANCE TO GOAL BY " + str(distance_difference))
            #    reward += abs(distance_difference/self.desired_point.x)
            #if distance_from_des_point < 1.0:
            #    reward = 0.01/distance_from_des_point
            

        else:
            if self.is_in_desired_position(current_position, epsilon=0.3):
                reward = self.end_episode_points
            else:
                reward = -0.25*self.end_episode_points

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods

    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False

        discretized_ranges = []
        mod = len(data.ranges)/new_ranges

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))

                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))


        return discretized_ranges


    #def get_vector_magnitude(self, vector):
    #    """
    #    It calculated the magnitude of the Vector3 given.
    #    This is usefull for reading imu accelerations and knowing if there has been
    #    a crash
    #    :return:
    #    """
    #    contact_force_np = numpy.array((vector.x, vector.y, vector.z))
    #    force_magnitude = numpy.linalg.norm(contact_force_np)
    #
    #    return force_magnitude


    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        x_curr = current_position.x
        y_curr = current_position.y
        x_goal = self.desired_point.x
        y_goal = self.desired_point.y

        distance = math.sqrt(pow((x_curr - x_goal),2) + pow((y_curr - y_goal),2))

        return distance


    def is_in_desired_position(self,current_position, epsilon=0.1):
        """
        It returns True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False


        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos

    
    def get_distance_from_pedestrian(self, current_position, pedestrian):
        """
        Returns the distance from the robot to the cloest pedestrian
        """
        #x_current = current_position.x
        #y_current = current_position.y

        # pedestrian = [relative x_coordinate, relative y_coordinate, yaw_angle, speed]
        ped_x = pedestrian[0]
        ped_y = pedestrian[1]

        #distance = math.sqrt(pow((x_current - ped_x),2) + pow((y_current - ped_y),2))
        distance = math.sqrt(pow(ped_x,2) + pow(ped_y,2))
        
        return distance


    def get_valid_odom(self, max_retries = 20):
        """
        Returns a valid odometry reading
        """
        odometry = self.get_odom()
        i = 0
        while odometry is None and i < max_retries:
            odometry = self.get_odom()
            rospy.logwarn("TRYING TO GET ODOM: " + str(odometry))
            i += 1

        return odometry

    def discretize_yaw(self, orientation):
        """
        Returns discretized yaw angle from orientation:
        0 = NORTH
        1 = EAST
        2 = SOUTH
        3 = WEST
        """
        quaternion = (orientation.x,
                      orientation.y,
                      orientation.z,
                      orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        discretized_yaw = 0

        if (yaw <= 45*math.pi/180) and (yaw >= -45*math.pi/180):
            discretized_yaw = 0
        elif (yaw <= -45*math.pi/180) and (yaw >= -135*math.pi/180):
            discretized_yaw = 1
        elif (yaw <= -135*math.pi/180) or (yaw >= 135*math.pi/180):
            discretized_yaw = 2
        elif (yaw <= 135*math.pi/180) and (yaw >= 45*math.pi/180):
            discretized_yaw = 3
        else:
            rospy.logerr("COULD NOT GET DISCRETIZED ROBOT ORIENTATION!" + str(yaw*180/math.pi))

        return int(discretized_yaw)


    def has_crashed_scan(self,data):
        """
        Checks if robot has crashed based on laserscan readings
        """
        self._episode_done = False
        rospy.logdebug("data=" + str(data))

        for i,item in enumerate(data.ranges):
            if (self.min_range > item > 0):
                rospy.logdebug("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                self._episode_done = True
            else:
                rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))

        return self._episode_done

    def round_to_base(self, number, base):
        multiplier = 1/base
        return (round(number*multiplier)/multiplier)
