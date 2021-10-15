#!/usr/bin/env python

import rospy
import tf
import math
import geometry_msgs.msg
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class control_actor:
    def __init__(self):
        self.odometry_sub = rospy.Subscriber("burger/odom", Odometry, self.odometry_callback)
        self.start_y = rospy.get_param('~start_y', '0')
        self.goal_y = rospy.get_param('~goal_y', '0')
        self.cmd_pub = rospy.Publisher("burger/cmd_vel", geometry_msgs.msg.Twist, queue_size=10)
        self.marker_pub   = rospy.Publisher("robot_visualization", Marker, queue_size = 1)
        self.speed = 0.40

    def odometry_callback(self, odometry):
        msg = geometry_msgs.msg.Twist()
        current_pose = odometry.pose
        current_y = current_pose.pose.position.y

        if self.start_y < self.goal_y:
            if current_y < self.goal_y:
                msg.linear.x = -1*self.speed
            else:
                self.start_y = -1*self.start_y
                self.goal_y = -1*self.goal_y
                msg.linear.x = self.speed
            current_pose.pose.orientation = self.change_orientation(current_pose.pose.orientation, math.pi)

        elif self.start_y > self.goal_y:
            if current_y > self.goal_y:
                msg.linear.x = self.speed
            else:
                self.start_y = -1*self.start_y
                self.goal_y = -1*self.goal_y
                msg.linear.x = -1*self.speed

        else:
            rospy.logwarn("Start and goal position of actors should not be equal")

        self.cmd_pub.publish(msg)
        self.publish_rviz_marker(current_pose)

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

    def publish_rviz_marker(self, pose):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp    = rospy.get_rostime()
        marker.type     = marker.ARROW
        marker.action   = marker.ADD
        marker.scale.x  = 0.25
        marker.scale.y  = 0.2
        marker.scale.z  = 0.25
        marker.pose.position.x  = pose.pose.position.x
        marker.pose.position.y  = pose.pose.position.y - np.sign(self.goal_y)*marker.scale.y/2
        marker.pose.position.z  = 0.25
        marker.pose.orientation = pose.pose.orientation
        marker.color.a = 0.5
        marker.color.r = 0.0    
        marker.color.g = 1.0    
        marker.color.b = 0.0  
        self.marker_pub.publish(marker)


if __name__ == '__main__':
    rospy.init_node('control_actor', anonymous=False)
    control_actor()
    rospy.spin()
