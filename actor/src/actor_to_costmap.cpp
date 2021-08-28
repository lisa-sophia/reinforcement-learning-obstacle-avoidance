#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/PoseStamped.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>

costmap_2d::Costmap2DROS* costmap_ros;
//costmap_2d::Costmap2D* costmap;

void actorStateCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    geometry_msgs::PoseStamped actor_position = geometry_msgs::PoseStamped();
    actor_position.pose = msg->pose;
    //costmap = costmap_ros->getCostmap();
    unsigned int mx, my;
    costmap_ros->getCostmap()->worldToMap(actor_position.pose.position.x, actor_position.pose.position.y, mx, my);
    costmap_ros->getCostmap()->setCost(mx, my, costmap_2d::LETHAL_OBSTACLE);
    costmap_ros->updateMap();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "actor_to_costmap");
    ros::NodeHandle n;

    ros::Subscriber actor_sub = n.subscribe("actor_1/state", 1000, actorStateCallback);
    tf2_ros::Buffer tf_buffer;
    tf_buffer.setUsingDedicatedThread(true);
    //tf2_ros::TransformListener tf2_listener(tf_buffer);
    costmap_ros = new costmap_2d::Costmap2DROS("actor_costmap", tf_buffer);
    costmap_ros->start();

    ros::spin();
    return 0;
}