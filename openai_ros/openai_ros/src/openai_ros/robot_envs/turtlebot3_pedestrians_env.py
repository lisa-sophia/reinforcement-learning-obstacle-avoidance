from math import sqrt
import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from rospy.core import logwarn
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from openai_ros.openai_ros_common import ROSLauncher
from math import pi
import tf


class TurtleBot3PedEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new TurtleBot3Env environment.
        TurtleBot3 doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /imu: Inertial Mesuring Unit that gives relative accelerations and orientations.
        * /scan: Laser Readings

        Actuators Topic List: /cmd_vel,

        Args:
        """
        rospy.logdebug("Start TurtleBot3PedEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name="simulation",
                    launch_file_name="put_robot_in_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = [] #["imu"]

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TurtleBot3PedEnv, self).__init__(controllers_list=self.controllers_list,
                                               robot_name_space=self.robot_name_space,
                                               reset_controls=False,
                                               start_init_physics_parameters=False)




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        #rospy.Subscriber("/imu", Imu, self._imu_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)

        # also subscribe to the pedestrians / agents
        rospy.Subscriber("/actor_states", PoseArray, self._actor_states_callback)
        rospy.Subscriber("/actor_speeds", Float32MultiArray, self._actor_speeds_callback)
        self.closest_ped = None
        self.closest_ped_state = Pose()
        self.closest_ped_state.position.x = None
        self.closest_ped_state.position.y = None
        self.closest_ped_state.position.z = None
        self.closest_ped_state.orientation.x = None
        self.closest_ped_state.orientation.y = None
        self.closest_ped_state.orientation.z = None
        self.closest_ped_state.orientation.w = None
        self.closest_ped_speed = None

        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()

        rospy.logdebug("Finished TurtleBot3PedEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------


    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        #self._check_imu_ready()
        self._check_laser_scan_ready()
        self._check_actors_states_ready()
        self._check_actors_speeds_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /odom READY=>")

            except:
                rospy.logerr("Current /odom not ready yet, retrying for getting odom")

        return self.odom


    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/imu", Imu, timeout=5.0)
                rospy.logdebug("Current /imu READY=>")

            except:
                rospy.logerr("Current /imu not ready yet, retrying for getting imu")

        return self.imu


    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan


    def _check_actors_states_ready(self):
        self.actor_states = None
        rospy.logdebug("Waiting for /actor_states to be READY...")
        while self.actor_states is None and not rospy.is_shutdown():
            try:
                self.actor_states = rospy.wait_for_message("/actor_states", PoseArray, timeout=1.0)
                rospy.logdebug("Current /actor_states READY=>")

            except:
                rospy.logerr("Current /actor_states not ready yet, retrying for getting actor_states")
        return self.actor_states


    def _check_actors_speeds_ready(self):
        self.actor_speeds = None
        rospy.logdebug("Waiting for /actor_speeds to be READY...")
        while self.actor_speeds is None and not rospy.is_shutdown():
            try:
                self.actor_speeds = rospy.wait_for_message("/actor_speeds", Float32MultiArray, timeout=1.0)
                rospy.logdebug("Current /actor_speeds READY=>")

            except:
                rospy.logerr("Current /actor_speeds not ready yet, retrying for getting actor_speeds")
        return self.actor_speeds


    def _odom_callback(self, data):
        self.odom = data

    def _imu_callback(self, data):
        self.imu = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _actor_states_callback(self, data):
        # the goal is in positive x-direction of the robot, so we only check the pedestrians in front of us
        # goal is at (5,0), start position is (0,0), pedestrians are in-between
        self.closest_ped = None
        self.closest_ped_state.position.x = None
        self.closest_ped_state.position.y = None
        self.closest_ped_state.position.z = None
        self.closest_ped_state.orientation.x = None
        self.closest_ped_state.orientation.y = None
        self.closest_ped_state.orientation.z = None
        self.closest_ped_state.orientation.w = None

        current_odometry = self.get_odom()
        if current_odometry is None:
            rospy.logdebug("Odometry not ready yet, will not get pedestrian info yet")
            return

        curr_x = current_odometry.pose.pose.position.x
        curr_y = current_odometry.pose.pose.position.y
        # we do not care about actors that are more than 3 metres away, since robot doesn't move that fast
        min_dist = 3.0
        i = 0
        for p in data.poses:
            actor_x = p.position.x
            actor_y = p.position.y
            dist = sqrt(pow(actor_x-curr_x, 2) + pow(actor_y-curr_y, 2))
            if dist < min_dist: #and actor_x > (curr_x - 1): # this could be a future improvement
                self.closest_ped = i
                min_dist = dist
                self.closest_ped_state = p
            i += 1

        rospy.logdebug("CLOSEST PEDESTRIAN = Nr. " + str(self.closest_ped) 
                        + "WITH STATE = " + str(self.closest_ped_state))


    def _actor_speeds_callback(self, data):
        if self.closest_ped is None:
            self.closest_ped_speed = None
        else:
            self.closest_ped_speed = data.data[self.closest_ped]
        rospy.logdebug("CLOSEST PEDESTRIAN SPEED = " + str(self.closest_ped_speed))
        

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        #self.wait_until_twist_achieved(cmd_vel_value,epsilon,update_rate)
        # We place a wait of certain amount of time, because the twist achieved function doesnt work properly
        # -> give robot a bit more time to perform turning maneuver
        if (angular_speed < 0.01):
            time.sleep(0.3) #time.sleep(0.5)
        else:
            time.sleep(0.5) #time.sleep(1.0)

    def custom_move_base(self, start_pose, action, linear_speed, linear_turn_speed, angular_speed, step_size, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        if action == "FORWARDS": 
            # move one step in positive x-direction
            self.change_position(start_pose, step_size, 0.0, 0.0)

        elif action == "TURN_LEFT": 
            # move one step in positive y-direction
            self.change_position(start_pose, 0.0, step_size, 90.0)

        elif action == "TURN_RIGHT": 
            # move one step in negative y-direction
            self.change_position(start_pose, 0.0, -1*step_size, -90.0)

        elif action == "STOP": 
            # stand still
            cmd_vel_value = Twist()
            cmd_vel_value.linear.x = 0.0
            cmd_vel_value.angular.z = 0.0
            rospy.logdebug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
            self._check_publishers_connection()
            self._cmd_vel_pub.publish(cmd_vel_value)
            time.sleep(1.0) # how long??

    def change_position(self, start_odom, x_step, y_step, desired_yaw):
        """
        It will move the base according to the given steps in x and y direction.
        It will wait until those changes are achived reading from the odometry topic.
        :param start_odom: 
        :param x_step: 
        :param y_step:
        :param desired_yaw: 
        :return:
        """       
        yaw_epsilon = 0.5 # in degrees, not rad
        position_epsilon = 0.05 # in metres
        rate = rospy.Rate(10) # 10 Hz

        # abort if turning takes too long
        start_time_action = rospy.get_rostime().to_sec()
        max_action_time   = 5.0

        # first check that correct orientation is achieved
        while not rospy.is_shutdown():
            current_odom = self._check_odom_ready()
            current_quat = (current_odom.pose.pose.orientation.x,
                            current_odom.pose.pose.orientation.y,
                            current_odom.pose.pose.orientation.z,
                            current_odom.pose.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(current_quat)
            current_yaw = euler[2]*180/pi
            delta_yaw = desired_yaw - current_yaw
            if abs(delta_yaw) < yaw_epsilon:
                rospy.logwarn("==> Achieved desired yaw angle.")
                cmd_vel_value = Twist()
                cmd_vel_value.linear.x = 0.0
                cmd_vel_value.angular.z = 0.0
                self._check_publishers_connection()
                self._cmd_vel_pub.publish(cmd_vel_value)
                rate.sleep()
                break
            else:
                rospy.loginfo("Not achieved desired yaw = " + str(desired_yaw) + " yet ; current yaw = " + str(current_yaw))
                cmd_vel_value = Twist()
                cmd_vel_value.linear.x = 0.0
                cmd_vel_value.angular.z = -1 * numpy.sign(delta_yaw) * 1.0
                if abs(delta_yaw) < 10:
                    cmd_vel_value.angular.z = -1 * numpy.sign(delta_yaw) * 0.2
                rospy.logdebug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
                self._check_publishers_connection()
                self._cmd_vel_pub.publish(cmd_vel_value)
                rate.sleep()
            if (rospy.get_rostime().to_sec() - start_time_action) > max_action_time:
                rospy.logerr("ABORTING ACTION: Took longer than " + str(max_action_time) + " to execute.")
                break

        # abort if position change takes too long
        start_time_action = rospy.get_rostime().to_sec()
        max_action_time   = 2.0

        # now check that correct position change is achieved
        while not rospy.is_shutdown():
            current_odom = self._check_odom_ready()
            current_x = current_odom.pose.pose.position.x
            current_y = current_odom.pose.pose.position.y
            desired_x = start_odom.pose.pose.position.x + x_step
            desired_y = start_odom.pose.pose.position.y + y_step

            delta_x = desired_x - current_x
            delta_y = desired_y - current_y
            
            # if we want to move forward and have achieved desired change
            if x_step > 0.01 and abs(delta_x) < position_epsilon:
                rospy.logwarn("==> Achieved desired x-position.")
                cmd_vel_value = Twist()
                cmd_vel_value.linear.x = 0.0
                cmd_vel_value.angular.z = 0.0
                self._check_publishers_connection()
                self._cmd_vel_pub.publish(cmd_vel_value)
                rate.sleep()
                break
            # if we want to move left/right and have achieved desired change
            elif abs(y_step) > 0.01 and abs(delta_y) < position_epsilon:
                rospy.logwarn("==> Achieved desired y-position.")
                cmd_vel_value = Twist()
                cmd_vel_value.linear.x = 0.0
                cmd_vel_value.angular.z = 0.0
                self._check_publishers_connection()
                self._cmd_vel_pub.publish(cmd_vel_value)
                rate.sleep()
                break
            else:
                rospy.loginfo("Not achieved position change yet, keep going.")
                cmd_vel_value = Twist()
                cmd_vel_value.linear.x = 0.3
                cmd_vel_value.angular.z = 0.0
                rospy.logdebug("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
                self._check_publishers_connection()
                self._cmd_vel_pub.publish(cmd_vel_value)
                rate.sleep()
            if (rospy.get_rostime().to_sec() - start_time_action) > max_action_time:
                rospy.logerr("ABORTING ACTION: Took longer than " + str(max_action_time) + " to execute.")
                break

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.logdebug("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05

        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))

        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon

        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready()
            # IN turtlebot3 the odometry angular readings are inverted, so we have to invert the sign.
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = -1*current_odometry.twist.twist.angular.z

            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")

        rospy.logdebug("END wait_until_twist_achieved...")

        return delta_time


    def get_odom(self):
        return self.odom

    def get_imu(self):
        return self.imu

    def get_laser_scan(self):
        return self.laser_scan

    def get_closest_pedestrian_state(self):
        return self.closest_ped_state
    
    def get_closest_pedestrian_speed(self):
        return self.closest_ped_speed
