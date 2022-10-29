#! /usr/bin/env python3
# Copyright (c) 2022 Enrico Sutera
# Copyright (c) 2022 Samsung R&D Institute Russia
# Copyright (c) 2022 Joshua Wallace
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator
import rclpy
from rclpy.node import Node
import threading

from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory

import math
import os
import pickle
import glob
import time
import numpy as np
import math

from random import seed
from random import randint
from random import uniform

from transforms3d.euler import euler2quat, quat2euler


def getPlannerResults(odom_tf_broadcaster, navigator, initial_pose, goal_pose, planners):
    results = []
    # Publish tf with initialpose (specifically initialpose is not required, could be at origin )
    odom_to_baselink_tf_msg = TransformStamped()
    odom_to_baselink_tf_msg.header.frame_id = 'odom'
    odom_to_baselink_tf_msg.child_frame_id = 'base_link'
    rclpy.spin_once(odom_tf_broadcaster)
    odom_to_baselink_tf_msg.header.stamp = odom_tf_broadcaster.get_clock().now().to_msg()
    odom_to_baselink_tf_msg.transform.rotation.x = initial_pose.pose.orientation.x
    odom_to_baselink_tf_msg.transform.rotation.y = initial_pose.pose.orientation.y
    odom_to_baselink_tf_msg.transform.rotation.z = initial_pose.pose.orientation.z
    odom_to_baselink_tf_msg.transform.rotation.w = initial_pose.pose.orientation.w
    odom_to_baselink_tf_msg.transform.translation.x = initial_pose.pose.position.x
    odom_to_baselink_tf_msg.transform.translation.y = initial_pose.pose.position.y
    odom_to_baselink_tf_msg.transform.translation.z = initial_pose.pose.position.z
    odom_tf_broadcaster.tf_broadcaster.sendTransform(odom_to_baselink_tf_msg)

    for planner in planners:
        path = navigator._getPathImpl(initial_pose, goal_pose, planner, use_start=True)
        if path is not None:
            results.append(path)
        else:
            return results
    return results

def getRandomStart(costmap, max_cost, side_buffer, time_stamp, res):
    start = PoseStamped()
    start.header.frame_id = 'map'
    start.header.stamp = time_stamp
    while True:
        row = randint(side_buffer, costmap.shape[0]-side_buffer)
        col = randint(side_buffer, costmap.shape[1]-side_buffer)

        if costmap[row, col] < max_cost:
            start.pose.position.x = col*res
            start.pose.position.y = row*res

            yaw = uniform(0, 1) * 2*math.pi
            quad = euler2quat(0.0, 0.0, yaw)
            start.pose.orientation.w = quad[0]
            start.pose.orientation.x = quad[1]
            start.pose.orientation.y = quad[2]
            start.pose.orientation.z = quad[3]
            break
    return start

def getRandomGoal(costmap, start, max_cost, side_buffer, time_stamp, res):
    goal = PoseStamped()
    goal.header.frame_id = 'map'
    goal.header.stamp = time_stamp
    while True:
        row = randint(side_buffer, costmap.shape[0]-side_buffer)
        col = randint(side_buffer, costmap.shape[1]-side_buffer)

        start_x = start.pose.position.x
        start_y = start.pose.position.y
        goal_x = col*res
        goal_y = row*res
        x_diff = goal_x - start_x
        y_diff = goal_y - start_y
        dist = math.sqrt(x_diff ** 2 + y_diff ** 2)

        if costmap[row, col] < max_cost and dist > 3.0:
            goal.pose.position.x = goal_x
            goal.pose.position.y = goal_y

            yaw = uniform(0, 1) * 2*math.pi
            quad = euler2quat(0.0, 0.0, yaw)
            goal.pose.orientation.w = quad[0]
            goal.pose.orientation.x = quad[1]
            goal.pose.orientation.y = quad[2]
            goal.pose.orientation.z = quad[3]
            break
    return goal

def SimulateMovement(pose, twist, frequency):
    # Basically compute odometry from twist since there's no localization

    _,_,yaw = quat2euler([pose.pose.orientation.w, pose.pose.orientation.x,
                        pose.pose.orientation.y, pose.pose.orientation.z])

    delta_yaw = twist.angular.z  * 1.0/frequency
    delta_x = twist.linear.x * math.cos(yaw + delta_yaw) * 1.0/frequency
    delta_y = twist.linear.x * math.sin(yaw + delta_yaw) * 1.0/frequency
    pose.pose.position.x += delta_x
    pose.pose.position.y += delta_y


    yaw = yaw + delta_yaw
    quad = euler2quat(0.0, 0.0, yaw)
    pose.pose.orientation.w = quad[0]
    pose.pose.orientation.x = quad[1]
    pose.pose.orientation.y = quad[2]
    pose.pose.orientation.z = quad[3]
    #print("vel x ", twist.linear.x, "vel theta", twist.angular.z)
    #print("delta x ", delta_x, "delta theta", delta_yaw)
    #print("pose -> x: ", pose.pose.position.x, "y: ", pose.pose.position.y, "yaw: ", yaw)
    return pose

class CmdVelListener(Node):

    def __init__(self):
        super().__init__('benchmark_cmd_vel_node',
            cli_args= ['--ros-args','-p', 'use_sim_time:=True' ])
        self.twist_msg = Twist()
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused variable warning

    def cmd_vel_callback(self, msg):
        self.twist_msg = msg

class OdomPublisher(Node):

    def __init__(self):
        super().__init__('odom_publisher',
            cli_args= ['--ros-args','-p', 'use_sim_time:=True' ])
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)

class OdomBroadcaster(Node):

    def __init__(self):
        super().__init__('odom_broadcaster',
            cli_args= ['--ros-args','-p', 'use_sim_time:=True' ])
                        #parameter_overrides = [rclpy.parameter.Parameter('use_sim_time',type_=rclpy.parameter.Parameter.Type.BOOL , value=True)])
        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

def getControllerResults(odom_tf_broadcaster, navigator, path, controllers, pose, controller_frequency):

    cmd_vel_subscriber_node = CmdVelListener()

    odom_node = OdomPublisher()
    odom_msg = Odometry()
    odom_msg.header.frame_id = 'odom'
    odom_msg.child_frame_id = 'base_link'
    
    odom_to_baselink_tf_msg = TransformStamped()
    odom_to_baselink_tf_msg.header.frame_id = 'odom'
    odom_to_baselink_tf_msg.child_frame_id = 'base_link'
    rclpy.spin_once(odom_node)
    odom_to_baselink_tf_msg.header.stamp = odom_node.get_clock().now().to_msg()

    for controller in controllers:
        print("Getting controller: ", controller)
        i = 0
        navigator.followPath(path.path, controller)
        while not navigator.isTaskComplete():
            # feedback does not provide both linear and angular
            # we get velocities from cmd_vel
            rclpy.spin_once(cmd_vel_subscriber_node)

            # Update "virtual" pose considering twist 
            pose = SimulateMovement(pose,cmd_vel_subscriber_node.twist_msg,controller_frequency)

            # Provide odom to controller for next loop
            odom_msg.pose.pose = pose.pose
            odom_msg.twist.twist = cmd_vel_subscriber_node.twist_msg
            rclpy.spin_once(odom_node)
            odom_msg.header.stamp = odom_node.get_clock().now().to_msg()
            odom_node.odom_pub.publish(odom_msg)

            # Update tf for next loop
            odom_to_baselink_tf_msg.header.stamp = odom_node.get_clock().now().to_msg()
            odom_to_baselink_tf_msg.transform.rotation.x = odom_msg.pose.pose.orientation.x
            odom_to_baselink_tf_msg.transform.rotation.y = odom_msg.pose.pose.orientation.y
            odom_to_baselink_tf_msg.transform.rotation.z = odom_msg.pose.pose.orientation.z
            odom_to_baselink_tf_msg.transform.rotation.w = odom_msg.pose.pose.orientation.w
            odom_to_baselink_tf_msg.transform.translation.x = odom_msg.pose.pose.position.x
            odom_to_baselink_tf_msg.transform.translation.y = odom_msg.pose.pose.position.y
            odom_to_baselink_tf_msg.transform.translation.z = odom_msg.pose.pose.position.z
            odom_tf_broadcaster.tf_broadcaster.sendTransform(odom_to_baselink_tf_msg)
            # Do something with the feedback
            # TODO metrics should be gathered here

    cmd_vel_subscriber_node.destroy_node()

def main():
    rclpy.init()

    time.sleep(4)
    navigator = BasicNavigator()

    odom_tf_broadcaster = OdomBroadcaster()

    odom_to_baselink_tf_msg = TransformStamped()
    odom_to_baselink_tf_msg.header.frame_id = 'odom'
    odom_to_baselink_tf_msg.child_frame_id = 'base_link'
    rclpy.spin_once(odom_tf_broadcaster)
    odom_to_baselink_tf_msg.header.stamp = odom_tf_broadcaster.get_clock().now().to_msg()
    odom_tf_broadcaster.tf_broadcaster.sendTransform(odom_to_baselink_tf_msg)

    time_stamp = navigator.get_clock().now().to_msg()
    print(time_stamp)
    
    # Wait for planner and controller to fully activate
    print("Waiting for planner and controller servers to activate")
    #navigator.waitUntilNav2Active('planner_server', 'controller_server')

    # Set map to use, other options: 100by100_15, 100by100_10
    map_path = os.getcwd() + '/' + glob.glob('**/100by100_20.yaml', recursive=True)[0]
    navigator.changeMap(map_path)
    time.sleep(2)   

    # Get the costmap for start/goal validation
    costmap_msg = navigator.getGlobalCostmap()
    costmap = np.asarray(costmap_msg.data)
    costmap.resize(costmap_msg.metadata.size_y, costmap_msg.metadata.size_x)

    planners = ['GridBased']
    controllers = ['FollowPath']

    controller_frequency = 20 #Hz
    max_cost = 210
    side_buffer = 100
    rclpy.spin_once(navigator)
    time_stamp = navigator.get_clock().now().to_msg()
    print(time_stamp)
    results = []
    seed(33)

    random_pairs = 10
    res = costmap_msg.metadata.resolution
    i = 0
    while len(results) != random_pairs:
        print("Cycle: ", i, "out of: ", random_pairs)
        start = getRandomStart(costmap, max_cost, side_buffer, time_stamp, res)
        goal = getRandomGoal(costmap, start, max_cost, side_buffer, time_stamp, res)

        print("Start", start)
        print("Goal", goal)

        result = getPlannerResults(odom_tf_broadcaster, navigator, start, goal, planners)
        if len(result) == len(planners):
            results.append(result)
        else:
            print("One of the planners was invalid")
            continue

        getControllerResults(odom_tf_broadcaster, navigator, result[0], controllers, start, controller_frequency)
        i = i +1

    print("Write Results...")

    # TODO save results once we have them
    print("Write Complete")
    exit(0)


if __name__ == '__main__':
    main()
