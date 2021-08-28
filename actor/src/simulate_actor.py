#!/usr/bin/env python

import rospy
import tf
import tf2_ros
import math
import copy
import numpy as np
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32

class simulate_actor:
    def __init__(self):
        self.odometry_sub = rospy.Subscriber("/odom", Odometry, self.odometry_callback)
        self.state_update = rospy.Publisher("state", geometry_msgs.msg.PoseStamped, queue_size = 10)
        self.speed_update = rospy.Publisher("speed", Float32, queue_size = 10)
        self.marker_pub   = rospy.Publisher("/actor_visualization", Marker, queue_size = 1)
        self.tfBuffer     = tf2_ros.Buffer()
        self.tf_listener  = tf2_ros.TransformListener(self.tfBuffer)

        self.agent_frame = "base_link"
        self.namespace   = rospy.get_param('~namespace', rospy.get_namespace())
        self.colour      = rospy.get_param('~colour', 'none')
        self.speed       = rospy.get_param('~speed', 'default')
        self.step_size   = 0.0
        if self.speed == "slow":
            self.step_size = 0.002
        elif self.speed == "default":
            self.step_size = 0.004
        elif self.speed == "fast":
            self.step_size = 0.008
        else:
            self.step_size = 0.004
            rospy.logwarn("Entered invalid speed. Options are 'slow', 'default' and 'fast'.")

        self.start_pose = geometry_msgs.msg.PoseStamped()
        self.start_pose.header.frame_id = "odom"
        self.start_pose.pose.position.x = rospy.get_param('~start_x', '0')
        self.start_pose.pose.position.y = rospy.get_param('~start_y', '0')
        self.start_pose.pose.position.z = 0.0
        quat = tf.transformations.quaternion_from_euler(0, 0, rospy.get_param('~start_yaw', '0'))
        self.start_pose.pose.orientation.x = quat[0]
        self.start_pose.pose.orientation.y = quat[1] 
        self.start_pose.pose.orientation.z = quat[2]
        self.start_pose.pose.orientation.w = quat[3]

        self.goal_pose = geometry_msgs.msg.PoseStamped()
        self.goal_pose.header.frame_id = "odom"
        self.goal_pose.pose.position.x = rospy.get_param('~goal_x', '0')
        self.goal_pose.pose.position.y = rospy.get_param('~goal_y', '0')
        self.goal_pose.pose.position.z = 0.0
        self.goal_pose.pose.orientation = copy.deepcopy(self.start_pose.pose.orientation)

        self.current_pose = geometry_msgs.msg.PoseStamped()
        self.current_pose.header.frame_id = "odom"
        self.current_pose.pose = copy.deepcopy(self.start_pose.pose)

        self.publish_pose = geometry_msgs.msg.PoseStamped()
        self.publish_pose.header.frame_id = self.agent_frame
        self.publish_pose.header.stamp = rospy.get_rostime()

    def odometry_callback(self, odometry):
        if self.start_pose.pose.position.y < self.goal_pose.pose.position.y:
            if self.current_pose.pose.position.y < self.goal_pose.pose.position.y:
                self.current_pose.pose.position.y += self.step_size
            else:
                # reached goal: turn around and drive to previous point
                self.reached_goal()
                self.current_pose.pose.position.y -= self.step_size

        elif self.start_pose.pose.position.y > self.goal_pose.pose.position.y:
            if self.current_pose.pose.position.y > self.goal_pose.pose.position.y:
                self.current_pose.pose.position.y -= self.step_size
            else:
                # reached goal: turn around and drive to previous point
                self.reached_goal()
                self.current_pose.pose.position.y += self.step_size

        else:
            rospy.logwarn("Start and goal position of actors should not be equal")

        msg = Float32()
        msg.data = self.step_size
        self.speed_update.publish(msg)

        self.current_pose.header.stamp = rospy.get_rostime()
        self.state_update.publish(self.current_pose)
        self.publish_rviz_marker()

    def change_orientation(self, original, yaw_change):
        q_rot  = tf.transformations.quaternion_from_euler(0, 0, yaw_change)
        q_orig = (original.x, original.y, original.z, original.w)
        q_new  = tf.transformations.quaternion_multiply(q_rot, q_orig)
        new    = geometry_msgs.msg.Quaternion()
        new.x  = q_new[0]
        new.y  = q_new[1]
        new.z  = q_new[2]
        new.w  = q_new[3]
        return new

    def reached_goal(self):
        tmp = self.start_pose.pose.position.y
        self.start_pose.pose.position.y = copy.deepcopy(self.goal_pose.pose.position.y)
        self.goal_pose.pose.position.y = copy.deepcopy(tmp)
        self.start_pose.pose.orientation = self.change_orientation(self.start_pose.pose.orientation, math.pi)
        self.goal_pose.pose.orientation = self.change_orientation(self.goal_pose.pose.orientation, math.pi)
        self.current_pose.pose.orientation = self.change_orientation(self.current_pose.pose.orientation, math.pi)

    def publish_rviz_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp    = rospy.get_rostime()
        marker.ns       = self.namespace
        marker.type     = marker.ARROW
        marker.action   = marker.ADD
        marker.scale.x  = 0.25
        marker.scale.y  = 0.2
        marker.scale.z  = 0.25
        marker.pose.position.x  = self.current_pose.pose.position.x 
        marker.pose.position.y  = self.current_pose.pose.position.y 
        marker.pose.position.z  = 0.25
        marker.pose.orientation = self.current_pose.pose.orientation
        marker.color = self.set_marker_colour()
        self.marker_pub.publish(marker)

    def set_marker_colour(self):
        col_marker = Marker()
        # transparency
        col_marker.color.a = 1.0 
        if self.colour == "red":
            col_marker.color.r = 1.0    
            col_marker.color.g = 0.0    
            col_marker.color.b = 0.0    
        elif self.colour == "green":
            col_marker.color.r = 0.0    
            col_marker.color.g = 1.0    
            col_marker.color.b = 0.0    
        elif self.colour == "blue":
            col_marker.color.r = 0.0    
            col_marker.color.g = 0.0    
            col_marker.color.b = 1.0   
        # default: white/grey marker
        else:   
            col_marker.color.r = 1.0   
            col_marker.color.g = 1.0    
            col_marker.color.b = 1.0   
        return col_marker.color

if __name__ == '__main__':
    rospy.init_node('simulate_actor', anonymous=False)
    simulate_actor()
    rospy.spin()
