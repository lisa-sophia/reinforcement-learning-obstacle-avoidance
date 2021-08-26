#!/usr/bin/env python

import rospy
import numpy as np
import geometry_msgs.msg
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray

class collect_actor_topics:
    def __init__(self):
        self.actor_states_pub = rospy.Publisher("actor_states", geometry_msgs.msg.PoseArray, queue_size = 1)
        self.actor_speeds_pub = rospy.Publisher("actor_speeds", Float32MultiArray, queue_size = 1)

        self.actor1_state = rospy.Subscriber("/actor_1/state", geometry_msgs.msg.PoseStamped, self._state1_callback)
        self.actor2_state = rospy.Subscriber("/actor_2/state", geometry_msgs.msg.PoseStamped, self._state2_callback)
        self.actor3_state = rospy.Subscriber("/actor_3/state", geometry_msgs.msg.PoseStamped, self._state3_callback)
        self.actor4_state = rospy.Subscriber("/actor_4/state", geometry_msgs.msg.PoseStamped, self._state4_callback)
        self.actor1_speed = rospy.Subscriber("/actor_1/speed", Float32, self._speed1_callback)
        self.actor2_speed = rospy.Subscriber("/actor_2/speed", Float32, self._speed2_callback)
        self.actor3_speed = rospy.Subscriber("/actor_3/speed", Float32, self._speed3_callback)
        self.actor4_speed = rospy.Subscriber("/actor_4/speed", Float32, self._speed4_callback)

        self.pose1  = geometry_msgs.msg.PoseStamped()
        self.pose2  = geometry_msgs.msg.PoseStamped()
        self.pose3  = geometry_msgs.msg.PoseStamped()
        self.pose4  = geometry_msgs.msg.PoseStamped()
        self.speed1 = Float32()
        self.speed2 = Float32()
        self.speed3 = Float32()
        self.speed4 = Float32()

        # bool list if state/speed has updated: state actor 1, speed actor 1, state actor 2, speed actor 2, etc...
        #self.has_changed = [False, False, False, False, False, False, False, False]
        self.has_changed = [False, False]

    def _state1_callback(self, data):
        self.pose1 = data.pose
        self.has_changed[0] = True
        self.publish_all()

    def _state2_callback(self, data):
        self.pose2 = data.pose
        self.has_changed[2] = True
        self.publish_all()

    def _state3_callback(self, data):
        self.pose3 = data.pose
        self.has_changed[4] = True
        self.publish_all()

    def _state4_callback(self, data):
        self.pose4 = data.pose
        self.has_changed[6] = True
        self.publish_all()


    def _speed1_callback(self, data):
        self.speed1 = data.data
        self.has_changed[1] = True
        self.publish_all()
        
    def _speed2_callback(self, data):
        self.speed2 = data.data
        self.has_changed[3] = True
        self.publish_all()

    def _speed3_callback(self, data):
        self.speed3 = data.data
        self.has_changed[5] = True
        self.publish_all()

    def _speed4_callback(self, data):
        self.speed4 = data.data
        self.has_changed[7] = True
        self.publish_all()
        

    def publish_all(self):
        for changed in self.has_changed:
            if changed is False:
                return
        
        states_msg = geometry_msgs.msg.PoseArray()
        states_msg.poses.append(self.pose1)
        #states_msg.poses.append(self.pose2)
        #states_msg.poses.append(self.pose3)
        #states_msg.poses.append(self.pose4)
        self.actor_states_pub.publish(states_msg)

        speeds_msg = Float32MultiArray()
        #speeds_msg.data = [self.speed1, self.speed2, self.speed3, self.speed4]
        speeds_msg.data = [self.speed1]
        self.actor_speeds_pub.publish(speeds_msg)

        for changed in self.has_changed:
            changed = False

if __name__ == '__main__':
    rospy.init_node('collect_actor_topics', anonymous=False)
    collect_actor_topics()
    rospy.spin()
