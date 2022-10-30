#! /usr/bin/env python3
# Copyright 2022 Joshua Wallace
# Copyright 2022 Enrico Sutera
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

import numpy as np
import math

import os
import pickle

from tabulate import tabulate
import nav_msgs.msg


def getPaths(results):
    paths = []
    for result in results:
        for path in result:
            paths.append(path.path)
    return paths

def getTaskAvgLinearSpeed(tasks_twists):
    averages_linear_x = []
    for task_twists in tasks_twists:
        for controller_twists in task_twists:
            linear_x = []
            for twist in controller_twists:
                linear_x.append(twist.linear.x)
            averages_linear_x.append(np.average(linear_x))
    return averages_linear_x

def getControllerPath(tasks_poses):
    controller_paths = []
    for task_poses in tasks_poses:
        for controller_poses in task_poses:
            path = nav_msgs.msg.Path()
            path.poses = controller_poses
            controller_paths.append(path)
    return controller_paths

def getTaskTimes(tasks_poses):
    controller_task_times = []
    for task_poses in tasks_poses:
        for controller_poses in task_poses:
            controller_task_times.append(
                (controller_poses[-1].header.stamp.nanosec/1e09+controller_poses[-1].header.stamp.sec)
                -(controller_poses[0].header.stamp.nanosec/1e09+controller_poses[0].header.stamp.sec))
    return controller_task_times

def getPathLength(path):
    path_length = 0
    x_prev = path.poses[0].pose.position.x
    y_prev = path.poses[0].pose.position.y
    for i in range(1, len(path.poses)):
        x_curr = path.poses[i].pose.position.x
        y_curr = path.poses[i].pose.position.y
        path_length = path_length + math.sqrt((x_curr-x_prev)**2 + (y_curr-y_prev)**2)
        x_prev = x_curr
        y_prev = y_curr
    return path_length

def main():

    # Read data
    print("Read data")
    with open(os.getcwd() + '/tasks_controller_results.pickle', 'rb') as f:
        tasks_results = pickle.load(f)
        
    with open(os.getcwd() + '/tasks_controller_twists.pickle', 'rb') as f:
        tasks_twists = pickle.load(f)
        
    with open(os.getcwd() + '/tasks_controller_poses.pickle', 'rb') as f:
        tasks_poses = pickle.load(f)

    with open(os.getcwd() + '/controllers.pickle', 'rb') as f:
        controllers = pickle.load(f)

    with open(os.getcwd() + '/planner_results.pickle', 'rb') as f:
        planner_results = pickle.load(f)

    # Compute metrics
    
    # Planner path lenght
    paths = getPaths(planner_results)
    path_lengths = []

    for path in paths:
        path_lengths.append(getPathLength(path))
    path_lengths = np.asarray(path_lengths)
    total_paths = len(paths)

    #speeds = getSpeeds(task_twists)
    
    # Linear Speed
    speeds_x = getTaskAvgLinearSpeed(tasks_twists)
    speeds_x = np.asarray(speeds_x)
    speeds_x.resize((int(total_paths/len(controllers)), len(controllers)))
    speeds_x = np.transpose(speeds_x)
    
    # Controllet path lenght
    controller_paths = getControllerPath(tasks_poses)
    controller_paths_lenght = []
    for controller_path in controller_paths:
        controller_paths_lenght.append(getPathLength(controller_path))
    controller_paths_lenght = np.asarray(controller_paths_lenght)
    controller_paths_lenght.resize((int(total_paths/len(controllers)), len(controllers)))
    controller_paths_lenght = np.transpose(controller_paths_lenght)
    # Generate table
    planner_table = [['Controller',
                      'Success rate',
                      'Average linear speed (m/s)',
                      'Average controller path len (m) ',
                      'Average time taken(s)) ']]

    task_times = getTaskTimes(tasks_poses)
    task_times = np.asarray(task_times)
    task_times.resize((int(total_paths/len(controllers)), len(controllers)))
    task_times = np.transpose(task_times)
    
    for i in range(0, len(controllers)):
        planner_table.append([controllers[i],
                              np.sum(tasks_results[i]),
                              np.average(speeds_x[i]),
                              np.average(controller_paths_lenght[i]),
                              np.average(task_times[i])])
    
    # Visualize results
    print("Planned average len: ", np.average(path_lengths))
    print("Total number of tasks: ", len(tasks_results))
    print(tabulate(planner_table))

if __name__ == '__main__':
    main()
