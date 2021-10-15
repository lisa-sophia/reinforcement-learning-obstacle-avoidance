#!/usr/bin/env python

import rospy
import geometry_msgs.msg

class invert_cmd_vel:
    def __init__(self):
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", geometry_msgs.msg.Twist, queue_size = 1)
        self.cmd_vel_sub = rospy.Subscriber("/cmd_vel_move_base", geometry_msgs.msg.Twist, self._cmd_vel_callback)

    def _cmd_vel_callback(self, data):
        cmd_vel_msg = data
        cmd_vel_msg.angular.z = -1 * cmd_vel_msg.angular.z
        self.cmd_vel_pub.publish(cmd_vel_msg)
        
if __name__ == '__main__':
    rospy.init_node('invert_cmd_vel', anonymous=False)
    invert_cmd_vel()
    rospy.spin()
