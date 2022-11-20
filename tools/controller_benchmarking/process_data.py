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
FATAL_COST = 210

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
            for twist_stamped in controller_twists:
                linear_x.append(twist_stamped.twist.linear.x)
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

def getObstacleDistances(tasks_local_costmaps, local_costmap_resolution):
    print("local_costmap_resolution ", local_costmap_resolution)
    controller_obstacle_distances = []
    # for each task / navigation
    for task_local_costamps in tasks_local_costmaps:
        # for each controller
        for controller_local_costmaps in task_local_costamps:
            min_obst_dist = 1e10
            for local_costmap in controller_local_costmaps:
                """
                # Heat map
                plt.imshow( local_costmap, cmap = 'rainbow' , interpolation = 'bilinear')
                # Add Title
                plt.title( "Heat Map" )
                # Display
                plt.show()
                """
                if local_costmap.max()>=FATAL_COST:
                    # Look for obstacle
                    obstacles_indexes = np.where(local_costmap>FATAL_COST)
                    # Shift indexes to centre of costmap since they
                    # will be used to compute distance from there
                    obstacles_indexes_x=np.add(obstacles_indexes[0],-np.round(local_costmap.shape[0]))
                    obstacles_indexes_y=np.add(obstacles_indexes[1],-np.round(local_costmap.shape[1]))

                    for x, y in zip(obstacles_indexes_x,obstacles_indexes_y):
                        osbstacle_distance = math.sqrt(x**2 + y**2) * local_costmap_resolution
                        if osbstacle_distance < min_obst_dist:
                            #print("New min obstacle distance found: ",osbstacle_distance)
                            min_obst_dist = osbstacle_distance
            controller_obstacle_distances.append(min_obst_dist)
    return controller_obstacle_distances

def getControllerMSJerks(tasks_twists):
    linear_ms_jerks = []
    angular_ms_jerks = []
    for task_twists in tasks_twists:
        for controller_twists in task_twists:
            linear_x = []
            angular_z = []
            time_passed = 0.0
            for twist_stamped in controller_twists:
                linear_x.append(twist_stamped.twist.linear.x)
                angular_z.append(twist_stamped.twist.angular.z)
                time_passed+=twist_stamped.header.stamp.nanosec/1e09+twist_stamped.header.stamp.sec
            end = controller_twists[-1].header.stamp.nanosec/1e09+controller_twists[-1].header.stamp.sec
            start= controller_twists[0].header.stamp.nanosec/1e09+controller_twists[0].header.stamp.sec
            dt = (end -start)/len(controller_twists)
            print(dt)
            linear_acceleration_x = np.gradient(linear_x, dt)
            angular_acceleration_z = np.gradient(angular_z, dt)
            linear_jerk_x = np.gradient(linear_acceleration_x, dt)
            angular_jerk_z = np.gradient(angular_acceleration_z, dt)
            # Mean Squared jerk Wininger, Kim, & Craelius (2009) 
            ms_linear_jerk_x = 0
            for jerk in linear_jerk_x:
                ms_linear_jerk_x += jerk**2
            linear_ms_jerks.append(ms_linear_jerk_x)
            ms_angular_jerk_x = 0
            for jerk in angular_jerk_z:
                ms_angular_jerk_x += jerk**2
            angular_ms_jerks.append(ms_angular_jerk_x)
    return   linear_ms_jerks,angular_ms_jerks
            
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

    with open(os.getcwd() + '/local_costmaps.pickle', 'rb') as f:
        tasks_controller_local_costmaps = pickle.load(f)

    with open(os.getcwd() + '/local_costmap_resolution.pickle', 'rb') as f:
        local_costmap_resolution = pickle.load(f)

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
    
    # Distance from obstacles
    
    # Minimum distance
    controller_obstacles_distances = getObstacleDistances(tasks_controller_local_costmaps, local_costmap_resolution)
    controller_obstacles_distances = np.asarray(controller_obstacles_distances)
    controller_obstacles_distances.resize((int(total_paths/len(controllers)), len(controllers)))
    controller_obstacles_distances = np.transpose(controller_obstacles_distances)
    
    # Smoothness
    
    controller_ME_linear_jerk,controller_ME_angular_jerk = getControllerMSJerks(tasks_twists)
    
    controller_ME_linear_jerk = np.asarray(controller_ME_linear_jerk)
    controller_ME_linear_jerk.resize((int(total_paths/len(controllers)), len(controllers)))
    controller_ME_linear_jerk = np.transpose(controller_ME_linear_jerk)
    
    controller_ME_angular_jerk = np.asarray(controller_ME_angular_jerk)
    controller_ME_angular_jerk.resize((int(total_paths/len(controllers)), len(controllers)))
    controller_ME_angular_jerk = np.transpose(controller_ME_angular_jerk)
    
    # Generate table
    planner_table = [['Controller',
                      'Success'+
                      '\nrate',
                      'Average linear'+
                      '\nspeed (m/s)',
                      'Average controller'+
                      '\npath len (m) ',
                      'Average time taken(s)) ',
                      'Minimum distance (m)'+
                      '\nfrom obstacle',
                      'Avg integrated x jerk' +
                      '\n(m^2/s^6)',
                      'Avg integrated z jerk' +
                      '\n(m/s^6)']]

    task_times = getTaskTimes(tasks_poses)
    task_times = np.asarray(task_times)
    task_times.resize((int(total_paths/len(controllers)), len(controllers)))
    task_times = np.transpose(task_times)
    
    for i in range(0, len(controllers)):
        planner_table.append([controllers[i],
                              np.sum(tasks_results[i]),
                              np.average(speeds_x[i]),
                              np.average(controller_paths_lenght[i]),
                              np.average(task_times[i]),
                              np.min(controller_obstacles_distances[i]),
                              np.average(controller_ME_linear_jerk),
                              np.average(controller_ME_angular_jerk)])

    
    # Visualize results
    print("Planned average len: ", np.average(path_lengths))
    print("Total number of tasks: ", len(tasks_results))
    print(tabulate(planner_table))

if __name__ == '__main__':
    main()
