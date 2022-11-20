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

from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory

import math
import os
import pickle
import glob
import time
import numpy as np
import math
import copy 
from random import seed
from random import randint
from random import uniform

from transforms3d.euler import euler2quat, quat2euler


def getPlannerResults(navigator, initial_pose, goal_pose, planners):
    results = []
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

class CmdVelListener(Node):

    def __init__(self):
        super().__init__('benchmark_cmd_vel_node')
        self.twist_msg = Twist()
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused variable warning

    def cmd_vel_callback(self, msg):
        self.twist_msg = msg

class OdomListener(Node):

    def __init__(self):
        super().__init__('benchmark_odom_node')
        self.odom_msg = Odometry()
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.subscription  # prevent unused variable warning

    def odom_callback(self, msg):
        self.odom_msg = msg

class GazeboWorldReset(Node):

    def __init__(self):
        super().__init__('gazebo_world_reset')
        self.cli = self.create_client(Empty, 'reset_world')
        
    def reset_world(self):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Empty.Request()
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def getControllerResults(navigator, path, controllers, pose):
    # Initialize results
    task_twists = []
    task_poses = []
    task_controller_results = []
    task_local_costmaps = []

    cmd_vel_subscriber_node = CmdVelListener()
    odom_subscriber_node = OdomListener()
    gazebo_resetter =  GazeboWorldReset()
    gazebo_resetter.reset_world()
    for controller in controllers:
        
        print("Getting controller: ", controller)
        i = 0
        navigator.followPath(path.path)
        task_controller_twists = []
        task_controller_poses = []
        task_controller_local_costmaps = []
        
        while not navigator.isTaskComplete():
            # feedback does not provide both linear and angular
            # we get velocities from cmd_vel
            rclpy.spin_once(cmd_vel_subscriber_node)
            rclpy.spin_once(odom_subscriber_node)
            # Get the local costmap for future metrics
            costmap_msg = navigator.getLocalCostmap()
            costmap = np.asarray(costmap_msg.data)
            costmap.resize(costmap_msg.metadata.size_y, costmap_msg.metadata.size_x)

            # Update "virtual" pose considering twist 
            pose = PoseStamped()
            pose.pose = odom_subscriber_node.odom_msg.pose.pose
            pose.header.stamp = odom_subscriber_node.get_clock().now().to_msg()
            task_controller_poses.append(copy.deepcopy(pose))
            twist_stamped = TwistStamped()
            twist_stamped.header.stamp = cmd_vel_subscriber_node.get_clock().now().to_msg()
            twist_stamped.twist = cmd_vel_subscriber_node.twist_msg
            task_controller_twists.append(twist_stamped)
            task_controller_local_costmaps.append(costmap)
            
            # Do something with the feedback
        if (navigator.getResult() == TaskResult.SUCCEEDED):
            task_controller_results.append(True)
        elif (navigator.getResult() == TaskResult.FAILED):
            task_controller_results.append(False)
        else:
            print("Unexpected result: \n", navigator.getResult())
            task_controller_results.append(False)
        task_twists.append(task_controller_twists)
        task_poses.append(task_controller_poses)
        task_local_costmaps.append(task_controller_local_costmaps)
        gazebo_resetter.reset_world()
    cmd_vel_subscriber_node.destroy_node()
    odom_subscriber_node.destroy_node()
    gazebo_resetter.destroy_node()
    return task_controller_results, task_twists, task_poses, task_local_costmaps

def main():
    rclpy.init()

    # A sleep for ensuring the Nav2 part is up, before senfing the tf
    time.sleep(4)
    navigator = BasicNavigator()

    # Wait for planner and controller to fully activate
    print("Waiting for planner and controller servers to activate")
    navigator.waitUntilNav2Active('planner_server', 'controller_server')

    # Set map to use, other options: 100by100_15, 100by100_10
    map_path = os.getcwd() + '/' + glob.glob('**/25by25_empty.yaml', recursive=True)[0]

    navigator.changeMap(map_path)
    time.sleep(2)  

    # Get the costmap for start/goal validation
    costmap_msg = navigator.getGlobalCostmap()
    costmap = np.asarray(costmap_msg.data)
    costmap.resize(costmap_msg.metadata.size_y, costmap_msg.metadata.size_x)

    local_costmap_msg = navigator.getLocalCostmap()
    local_costmap_resolution = local_costmap_msg.metadata.resolution

    planners = ['GridBased']
    controllers = ['FollowPath']

    max_cost = 210
    side_buffer = 100
    time_stamp = navigator.get_clock().now().to_msg()
    planner_results = []
    # Will collect all controller all task data
    tasks_controller_results = []
    tasks_controller_twists = []
    tasks_controller_poses = []
    # List with  local costamap of each controller for each task
    tasks_controller_local_costmaps = []
    seed(33)

    random_pairs = 1
    res = costmap_msg.metadata.resolution
    i = 0
    while len(planner_results) != random_pairs:
        print("Cycle: ", i, "out of: ", random_pairs)
        #start = getRandomStart(costmap, max_cost, side_buffer, time_stamp, res)
        start = PoseStamped()
        start.header.frame_id = 'map'
        start.header.stamp = time_stamp
        start.pose.position.x = 1.0
        start.pose.position.y = 1.0
        goal = getRandomGoal(costmap, start, max_cost, side_buffer, time_stamp, res)

        print("Start", start)
        print("Goal", goal)
        time.sleep(2)  
        result = getPlannerResults(navigator, start, goal, planners)
        if len(result) == len(planners):
            planner_results.append(result)
        else:
            print("One of the planners was invalid")
            continue
        # Change back map, planner will no more use this. Local costmap uses this map
        # with obstalce info
        task_controller_results,task_twists,task_poses, task_local_costmaps = getControllerResults(navigator, result[0], controllers, start)
        tasks_controller_results.append(task_controller_results)
        tasks_controller_twists.append(task_twists)
        tasks_controller_poses.append(task_poses)
        tasks_controller_local_costmaps.append(task_local_costmaps)
        i = i +1

    print("Write Results...")
    with open(os.getcwd() + '/tasks_controller_results.pickle', 'wb+') as f:
        pickle.dump(tasks_controller_results, f, pickle.HIGHEST_PROTOCOL)
    
    with open(os.getcwd() + '/tasks_controller_twists.pickle', 'wb+') as f:
        pickle.dump(tasks_controller_twists, f, pickle.HIGHEST_PROTOCOL)
        
    with open(os.getcwd() + '/tasks_controller_poses.pickle', 'wb+') as f:
        pickle.dump(tasks_controller_poses, f, pickle.HIGHEST_PROTOCOL)

    with open(os.getcwd() + '/controllers.pickle', 'wb+') as f:
        pickle.dump(controllers, f, pickle.HIGHEST_PROTOCOL)

    with open(os.getcwd() + '/local_costmaps.pickle', 'wb+') as f:
        pickle.dump(tasks_controller_local_costmaps, f, pickle.HIGHEST_PROTOCOL)
        
    with open(os.getcwd() + '/local_costmap_resolution.pickle', 'wb+') as f:
        pickle.dump(local_costmap_resolution, f, pickle.HIGHEST_PROTOCOL)

    with open(os.getcwd() + '/planner_results.pickle', 'wb+') as f:
        pickle.dump(planner_results, f, pickle.HIGHEST_PROTOCOL)
    # TODO save results once we have them
    print("Write Complete")
    exit(0)


if __name__ == '__main__':
    main()
