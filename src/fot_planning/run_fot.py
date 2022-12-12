#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
This module contains a local planner to perform
low-level waypoint following based on PID controllers.
"""

from collections import deque
import rospy
import math
import numpy as np
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from obj_msgs.msg import Obj, ObjList

""" import library for FOT """
# from PythonRobotics.PathPlanning.FrenetOptimalTrajectory import frenet_optimal_trajectory as FOT
import fot as FOT
import time
from ctypes import*
import os

""" THESE ARE RELATED TO IMPORTING C get_frenet() """
libname = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "fot_cpp_1.so"))
print("libname: ",libname)
c_lib = CDLL(libname)


class frenet_coordinate(Structure):
    _fields_ = [('s', c_double),
                ('d', c_double)
                ]


class cartesian_coordinate(Structure):
    _fields_ = [('x', c_double),
                ('y', c_double),
                ('heading', c_double)
                ]

c_lib.get_frenet.restype = frenet_coordinate
c_lib.get_cartesian.restype = cartesian_coordinate
# c_lib.get_dist.restype = get_dist

""" -------------------------------------------------"""

class Obstacle:
    def __init__(self):
        self.id = -1 # actor id
        self.vx = 0.0 # velocity in x direction
        self.vy = 0.0 # velocity in y direction
        self.vz = 0.0 # velocity in z direction
        self.ros_transform = None # transform of the obstacle in ROS coordinate
        self.carla_transform = None # transform of the obstacle in Carla world coordinate
        self.bbox = None # Bounding box w.r.t ego vehicle's local frame

class GlobalWp:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.s = 0

class MyLocalPlanner():
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is
    generated on-the-fly. The low-level motion of the vehicle is computed by using two PID
    controllers, one is used for the lateral control and the other for the longitudinal
    control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    # MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param role_name: name of the actor
        :param opt_dict: dictionary of arguments with the following semantics:

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal
                                         PID controller
                                         {'K_P':, 'K_D':, 'K_I'}
        """
        self.target_route_point = None
        self._current_waypoint = None
        self._vehicle_controller = None
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._vehicle_yaw = None
        self._current_speed = None
        self._current_pose = None
        self._current_acceleration = None
        self._obstacles = []
        self._obstacles_bbox_center = []
        self._wp_follower = False
        # Attributes for FOT
        self._si = 0 
        self._si_d = 0
        self._si_dd = 0
        self._sf = 1000000
        self._sf_d = 0
        self._sf_dd = 0

        self._di = 0
        self._di_d = 0
        self._di_dd = 0
        self._df_d = 0
        self._df_dd = 0

        self._lane_no = 0
        # self._start = 0
        # self._finished = False
        self._mapx = []
        self._mapy = []
        self._maps = []
        self._obs = []
        self._target_speed = 25
        self._vehicle_following = False
        self._no_paths = False
        # self._junction_encountered_before = False
        # self._next_waypoint = False
        self._MAX_T = 1.4
        self._MIN_T = 1.3
        self._stop = False
        self._ego_vehicle = None
        self._lane_change =None
        self._obj_list = []
        # get world and map for finding actors and waypoints
        
        self._target_point_publisher = rospy.Publisher(
            "/next_target", PointStamped, queue_size=1)

        # self._frenet_path_publisher = rospy.Publisher(
        #     "/frenet_path", PointStamped, queue_size=10)

        self._acceptPaths_pub = rospy.Publisher("/accept_paths", MarkerArray, queue_size=1)
        self._rejectPaths_pub = rospy.Publisher("/reject_paths", MarkerArray, queue_size=1)
        self._chosenPath_publisher = rospy.Publisher("/chosen_path", MarkerArray, queue_size=1)
        self._ref_path_pub = rospy.Publisher("/ref_path", Marker, queue_size=1)
        self._target_speed_pub = rospy.Publisher("/Target_Velocity", Float64, queue_size=1)

        self.fot_current_speed_subscriber = rospy.Subscriber("/fot_current_speed", Float64,self.currentSpeedCallback)
        self.fot_current_acc_subscriber = rospy.Subscriber("/fot_current_acc", Float64,self.currentAccCallback)
        self.fot_current_lane_change_subscriber = rospy.Subscriber("/fot_lane_change", String,self.laneChangeCallback)
        self.fot_current_odom_subscriber = rospy.Subscriber("/fot_current_odom", Odometry,self.currentOdomCallback)
        self.fot_mapx_subscriber = rospy.Subscriber("/fot_mapx", Float64MultiArray,self.mapxCallback)
        self.fot_mapy_subscriber = rospy.Subscriber("/fot_mapy", Float64MultiArray,self.mapyCallback)
        self.fot_maps_subscriber = rospy.Subscriber("/fot_maps", Float64MultiArray,self.mapsCallback)
        self.fot_veh_list_subscriber = rospy.Subscriber("/fot_veh_list", ObjList, self.objCallback)
        # print("self._lane_change", self._lane_change)
        # if self._lane_change:
        #     print("STARTING FOT")
        #     self.run_step()

    def objCallback(self, msg):
        # for i in self._obj_list:
        #     self._obj_list.pop()
        for i in self._obstacles:
            self._obstacles.pop()
            
        obj_list = msg.objlist
        for obj in obj_list:
            self._obstacles.append(obj)
            # obj_x, obj_y = obj.pose.position.x, obj.pose.position.y
            # self._obj_list.append([obj_x, obj_y])
        # for i in self._obj_list:
            # print("OBJECT: x ", i[0]," OBJECT: y ", i[1])

    def currentSpeedCallback(self, msg):
        self._current_speed = msg.data
        # print("fot_current_speed: ", self._current_speed)

    def currentAccCallback(self, msg):
        self._current_acceleration = msg.data
        # print("\nfot_current_acc: ", self._current_acceleration)

    def laneChangeCallback(self, msg):
        self._lane_change = msg.data
        # print("\nfot_lane_change: ", self._lane_change)

    def currentOdomCallback(self, msg):
        self._current_odom = msg
        self._current_pose = msg.pose.pose
        
        odom = self._current_pose.orientation
        quaternion = (
            odom.x,
            odom.y,
            odom.z,
            odom.w
        )
        _, _, self._vehicle_yaw = euler_from_quaternion(quaternion)
        # print("fot_current_pose: ", self._current_pose)

        # print("self._maps: ", self._maps)
        if (~(np.all(self._maps,0) and np.all(self._mapy,0) and np.all(self._mapx,0) )):
            self.run_step()
        else:
            print("WAITING FOR GLOBAL TRAJECTORY")

    def mapxCallback(self, msg):
        self._mapx = np.zeros(len(msg.data)-1)
        for i in range(len(msg.data)-1):
            self._mapx[i] = msg.data[i]
        # print("fot_mapx: ", self._mapx)
        

    def mapyCallback(self, msg):
        self._mapy = np.zeros(len(msg.data)-1)
        for i in range(len(msg.data)-1):
            self._mapy[i] = msg.data[i]
        # print("fot_mapy: ", self._mapy)
        
    def mapsCallback(self, msg):
        self._maps = np.zeros(len(msg.data)-1)
        for i in range(len(msg.data)-1):
            self._maps[i] = msg.data[i]
        # print("fot_maps: ", self._maps)
          
    def check_ob_collision_path(self, point):   # , l_vector, r_vector
        point.z = point.z + 0.75
        # return
        # print("POINT: \n", point)
        # print(self._obj_list)
        for ob in self._obstacles:

            # if self.check_obstacle(point, ob):
            if self.fot_collision_check(point, ob):

                return True
        return False



    def fot_collision_check(self, point, obstacle):
        # center = [obstacle.ros_transform.position.x, obstacle.ros_transform.position.y]
        obs_center = [obstacle.pose.position.x, obstacle.pose.position.y]
        d = math.sqrt(((point.x - obs_center[0]) ** 2 + (point.y - obs_center[1]) ** 2))
        # print(d)
        COL_CHECK = 2.29

        # if i == 0:
        #     print("check if obstacle falls within")
        #     COL_CHECK = 1.85

        collision = d <= 2*COL_CHECK  

        if collision:
            # print("rejected due to collision")
            return True

        return False



    def follow_path(self, current_speed, s):
        D0 = 5
        T = 1
        # if self._obstacles:
        if self._obj_list:
            time1 = time.time()
            i = 0
            for ob in self._obstacles:
                ob_speed = np.sqrt(ob.velocity.x*ob.velocity.x + ob.velocity.y*ob.velocity.y)

                i = i + 1
                # ob_s, ob_d = FOT.get_frenet(ob.ros_transform.position.x, ob.ros_transform.position.y, self._mapx, self._mapy)
                p1 = c_lib.get_frenet(c_double(ob.pose.position.x), c_double(ob.pose.position.y), self._mapx.ctypes.data_as(c_void_p), self._mapy.ctypes.data_as(c_void_p), c_long(len(self._mapx)))
                ob_s, ob_d = p1.s, p1.d

                # if (ob_s, ob_d) == (False, False):
                #     ob_s, ob_d = (-100, -100)
                # # except IndexError():
                #     print("CANT GET NEXT WP, -> ignore obstacle")
                #     self._vehicle_following = False
                #     break

                # time3 = time.time()

                # ob_vertices = ob.bbox.get_world_vertices(ob.carla_transform)
                # ob_v = [[o.x,o.y] for count, o in enumerate(ob_vertices) if count%2 == 0]
                # # print("ob_v, ",ob_v)
                # ego_vertices = self._ego_vehicle.bbox.get_world_vertices(self._ego_vehicle.carla_transform)
                # ego_v = [[o.x,o.y] for count, o in enumerate(ego_vertices) if count%2 == 0]

                # # print("ob_vertices, ",ob_vertices)
                # # print("ego_vertices, ",ego_vertices)

                # if SAT.separating_axis_theorem(ego_v, ob_v):
                    
                #     print("About to collide, emergency break ego")
                #     self._stop = True
                # else:
                #     self._stop = False
                #     print("emergency break off")

                # time4 = time.time()
                # print("elapsed time in SAT {}; freq {}".format((time4 - time3), 1/(time4 - time3)))

                ##  IF THERE'S A CAR WITHIN THE RANGE
                # if (ob_s - s) > -5 and (ob_s - s) < 15 :

                ##  IF THERE'S A CAR IN THE SAME LANE AS EGO
                if abs(ob_d - self._di) < self._LANE_WIDTH/2:

                    ##  if single lane/no_paths/junction && lv present -> follow lv
                    if ((self.both_invasion or (self._no_paths and (not self._vehicle_following))) and ((ob_s - s) > 0 and (ob_s - s) < 15)):         
                        # self._sf = ob_s - (D0 - T * (ob.speed/3.6))
                        self._sf = ob_s - 5

                        print(self._sf)
                        print("ob.velocity: ",ob.velocity)
                        self._target_speed = ob_speed
                        print(self._target_speed)
                        self._sf_d = ob_speed
                        self._vehicle_following = True

                        # if (ob_s - s) > 0 and (ob_s - s) < 10 and (current_speed - ob.speed) > 5  :
                        if (current_speed - ob_speed) > 5  :
                            self._stop = True
                        elif self._sf < s + 4:
                            self._stop = True
                        else:
                            self._stop = False

                        self._MAX_T = 4.15
                        self._MIN_T = 4.1

                        print("FOLLOWING VEHICLE invasion\n", ob_s)
                        break
                    
                    ##  FREE ROAD AVAILABLE BUT CAR AHEAD

                    ##  Donot overtake if lv speed close to speed limit
                        ##  Use some kind of velocity info to get a sense of direction
                    elif (abs(ob_speed - 30) < 10) and ((ob_s - s) > 0 (ob_s - s) < 15):        

                        # self._sf = ob_s - (D0 - T * (ob.speed/3.6))
                        self._sf = ob_s - 5
                        print(self._sf)
                        self._target_speed = ob_speed
                        self._sf_d = ob_speed
                        if self._sf < s:
                            self._stop = True
                        else:
                            self._stop = False
                        self._vehicle_following = True
                        self._MAX_T = 4.15
                        self._MIN_T = 4.1

                        print("FOLLOWING VEHICLE overtake\n", ob_s)
                        break
                    
                    else:
                        # print("FREE ROAD")
                        self._target_speed = 25 -   10 * abs(abs(self._vehicle_yaw) - abs(self._yaw_road))/(2*np.pi)
                        # if self._target_speed < 0:
                        #     print("\n NEGATIVE TARGET SPEED {} \n".format(self._target_speed))
                        # self._target_speed = 30
                        self._sf_d = self._target_speed
                        self._vehicle_following = False
                        self._stop = False
                        self._MAX_T = 1.4
                        self._MIN_T = 1.3
            
                # # ## car too close behind
                # elif abs(ob_s - s) < 3 and self._stop != True:
                #     if current_speed < ob.speed :                            
                #         print("Car too close behind. Speed up. s: {}, ob_s: {} ".format(s, ob_s))
                #         self._target_speed = ob.speed + 10
                #         # if self._target_speed < 0:
                #         #     print("\n NEGATIVE TARGET SPEED {} \n".format(self._target_speed))
                #         # self._target_speed = 30
                #         self._sf_d = self._target_speed
                #         self._stop = False
                #         self._vehicle_following = False
                #         self._MAX_T = 1.0
                #         self._MIN_T = 0.9
                #         break
                elif self._no_paths and (not self._vehicle_following) and abs(ob_d) < self._LANE_WIDTH/2:
                    self._sf = ob_s - 5

                    print(self._sf)
                    self._target_speed = ob_speed
                    self._sf_d = ob_speed
                    self._vehicle_following = True

                    # if (ob_s - s) > 0 and (ob_s - s) < 10 and (current_speed - ob.speed) > 5  :
                    if (current_speed - ob_speed) > 5  :
                        self._stop = True
                    elif self._sf < s + 4:
                        self._stop = True
                    else:
                        self._stop = False

                    self._MAX_T = 4.15
                    self._MIN_T = 4.1

                    print("FOLLOWING VEHICLE invasion\n", ob_s)
                    break
                    
                else:
                    print("FREE ROAD")
                    self._target_speed = 25 -   10 * abs(abs(self._vehicle_yaw) - abs(self._yaw_road))/(2*np.pi)
                    # if self._target_speed < 0:
                    #     print("\n NEGATIVE TARGET SPEED {} \n".format(self._target_speed))
                    # self._target_speed = 30
                    self._sf_d = self._target_speed
                    self._vehicle_following = False
                    self._stop = False
                    self._MAX_T = 1.4
                    self._MIN_T = 1.3

            time2 = time.time()
            # print("elapsed time in forloops {}; freq {}, count{}".format((time2 - time1), 1/(time2 - time1), i))
        else:
            # print("FREE ROAD")
            self._target_speed = 25 -   10 * abs(abs(self._vehicle_yaw) - abs(self._yaw_road))/np.pi

            # self._target_speed = 30
            self._sf_d = self._target_speed
            self._vehicle_following = False
            self._MAX_T = 1.4
            self._MIN_T = 1.3

            

    def run_step(self):
        """
        Execute one step of local planning which involves running the longitudinal
        and lateral PID controllers to follow the waypoints trajectory.
        """
        # print("RUN_STEP")
        # self._mapx = mapx; self._mapy = mapy; self._maps = maps;

        # print("lane_left {}, lane_right {}".format(d_left, d_right))
        # print("time taken to convert to frenet lanemarking {}, freq {}".format(time2-time1,1/(time2 -time1)))
        
        # print("self.get_coordinate_lanemarking(current_pose.position)",lanemarking_coordinate)
        self._LANE_WIDTH = 2.5
        # d_left = current_lane_number*lane_width
        # d_right = current_lane_number*laner_width
        d_left = self._LANE_WIDTH/2
        d_right = -self._LANE_WIDTH/2

        
        if self._lane_change == "LEFT":
            self.right_invasion = True
            self.left_invasion = False
            self.both_invasion = False


            # print("d_right",d_right)
            # print(self._lane_change)
            # left_invasion = False
        # else:
        #     right_invasion = False

        elif self._lane_change == "RIGHT":
            self.left_invasion = True
            self.right_invasion = False
            self.both_invasion = False

            # print("d_left",d_left)
            # print(self._lane_change)

            # right_invasion = False
        elif self._lane_change == "BOTH":
            self.left_invasion = True
            self.right_invasion = False
            self.both_invasion = False

            # print("d_left",d_left)
            # print("d_right",d_right)
            # print(self._lane_change)

        elif self._lane_change == "KEEP":
            # print("d_left",d_left)
            # print("d_right",d_right)
            # print(self._lane_change)

            self.right_invasion = False
            self.left_invasion = False
            self.both_invasion = True



        # if self._current_waypoint.left_lane_marking.lane_change in ["NONE", "Left"]:
        #     self.right_invasion = True
        # else:
        #     self.left_invasion = False
        # if self._current_waypoint.right_lane_marking.lane_change in ["NONE", "Right"]:
        #     self.right_invasion = True
        # else:
        #     self.right_invasion = False


        x = self._current_pose.position.x
        y = self._current_pose.position.y

        # s,d = FOT.get_frenet(x, y, self._mapx, self._mapy)
        # time1 = time.time()

        p1 = c_lib.get_frenet(c_double(x), c_double(y), self._mapx.ctypes.data_as(c_void_p), self._mapy.ctypes.data_as(c_void_p), c_long(len(self._mapx)))
        s, d = p1.s, p1.d
        # print("s,",s,"d,",d)
        # s,d = FOT.get_frenet(x, y, self._mapx, self._mapy)

        # if d < (self._lane_no - 1) *0.5 and d > (self._lane_no - 1) * 1.5:
        #     self._lane_no = self._lane_no - 1

        # elif d > (self._lane_no + 1) *0.5 and d < (self._lane_no + 1) * 1.5:
        #     self._lane_no = self._lane_no + 1

        # elif self._lane_no < 0 and d > (self._lane_no - 1):
        #     self._lane_no = self._lane_no + 1

        
        # elif self._lane_no > 0 and d < (self._lane_no + 1):
        #     self._lane_no = self._lane_no - 1

        if d < 0:
            self._lane_no , _ = divmod(abs(d), self._LANE_WIDTH)
            self._lane_no = -self._lane_no
        if d >= 0:
            self._lane_no , _ = divmod(abs(d), self._LANE_WIDTH)

        # print("lane_no: ", self._lane_no)
        # p1 = c_lib.get_frenet(c_double(x), c_double(y), self._mapx.ctypes.data_as(c_void_p), self._mapy.ctypes.data_as(c_void_p), c_long(len(self._mapx)))
        # s, d = p1.s, p1.d
        # time2 = time.time()
        # print("current_s {}, current_d {}".format(s,d))
        # print("time taken to convert to frenet s,d{}, freq {}".format(time2-time1,1/(time2 -time1)))

        # if self._maps[-1] - s < 3:
        #     self._finished = True
        # print("s {}\n, d {}\n".format(s,d))
        # time1 = time.time()
        """x, y, self._yaw_road = FOT.get_cartesian(s, d, self._mapx, self._mapy, self._maps)
        print("x: ",x, " y: ",y)"""
        q1 = c_lib.get_cartesian(c_double(s), c_double(d), self._mapx.ctypes.data_as(c_void_p), self._mapy.ctypes.data_as(c_void_p), self._maps.ctypes.data_as(c_void_p), c_long(len(self._maps)))
        x, y, self._yaw_road = q1.x, q1.y, q1.heading
        # time2 = time.time()
        # print("time taken for cartesian {}, freq {}".format(time2-time1,1/(time2 - time1)))
        # yaw_i = yaw - yaw_road
        self._yaw_i = self._vehicle_yaw - self._yaw_road

        # print("si {}, di {}, self._sf{} self._sf_d {}, self._df_d {}".format(self._si,self._di, self._sf, self._sf_d, self._df_d))
        self._si = s 
        self._si_d = self._current_speed * np.cos(self._yaw_i)
        self._si_dd = self._current_acceleration * np.cos(self._yaw_i)


        self._di = d
        self._di_d = self._current_speed * np.sin(self._yaw_i)
        self._di_dd = self._current_acceleration * np.sin(self._yaw_i)
        # self._di_dd = 0
        # print("si: ",self._si, " di: ",self._di)
        # time1 = time.time()
        # if self._si != 0 and self._sf == 0:
        #     print("frenet trying to connect back to start\n ENDING SIMULATION")
        #     control = self._vehicle_controller.run_step(
        #     0, current_speed, current_pose, self.target_route_point)
        #     control.brake = 1.0
        #     # self._finished = True
        #     return control, True

        # print("self._si {}, self._di {}".format(self._si, self._di))
        fplist = FOT.calc_frenet_paths(self._si, self._si_d/3.6, self._si_dd, self._sf, self._sf_d/3.6, self._sf_dd, 
                                        self._di, self._di_d/3.6, self._di_dd, self._df_d/3.6, self._df_dd, self._LANE_WIDTH, self._vehicle_following, self._MAX_T, self._MIN_T)
        # time2 = time.time()
        # print("time taken to calculate \nfot {}, \nfreq{}".format(time2-time1,1/(time2-time1)))
# 
        # time1 = time.time()
        fplist = FOT.calc_global_paths(fplist, self._mapx, self._mapy, self._maps)
        # time2 = time.time()
        # print("time taken to calculate global path {}, freq{}".format(time2-time1,1/(time2-time1)))
        
        



        # # print("s1 {}\n, d {}\n".format(s,d))
        # time1 = time.time()
        # self.get_obstacles(current_pose.position, 30.0)
        # time2 = time.time()
        # print("time taken to get obstacle {}, freq{}".format(time2-time1,1/(time2-time1)))

        # time1 = time.time()
        self.follow_path(self._current_speed, s)
        # time2 = time.time()
        # print("time taken to follow path{}, freq{}".format(time2-time1,1/(time2-time1)))

        # print("start")
        # for i, ob in enumerate(self._obstacles):
            # print("ego.s{}, ego.d {}, i: {}, ob.s {} ob.d {}".format(s,d,i,ob.s,ob.d))
        acceptPaths = MarkerArray()
        rejectPaths = MarkerArray()

        rejectFlag = [False] * len(fplist)

        for idx, fp in enumerate(fplist):
            msg = Marker()
            msg.header.frame_id = "map"
            msg.header.stamp = rospy.get_rostime()
            msg.pose.orientation.w = 1.0
            msg.id = idx
            msg.type = Marker.LINE_STRIP
            msg.color.r, msg.color.g, msg.color.b, msg.color.a = 0.0, 128.0, 0.0, 2.0
            msg.scale.x, msg.scale.y, msg.scale.z = 0.05, 0.05, 0.05

            # if abs(fp.d[-1]) > 1.5*self._LANE_WIDTH:
            #     print("fp.d[-1]",fp.d[-1])

            #     rejectFlag[idx] = True

            if self.left_invasion:
                # d = [i for i in fp.d]
                # print("d {}, d_left {}".format(fp.d[-1],d_left+ self._lane_no * self._LANE_WIDTH))
                # if any([d > d_left and d > 0.5 for d in fp.d]):
                # if fp.d[-1] > d_left:
                if fp.d[-1] > d_left + self._lane_no * self._LANE_WIDTH:
                    rejectFlag[idx] = True

            elif self.right_invasion:
                # d = [i for i in fp.d]
                # print("d {}, d_right {}".format(fp.d[-1],d_right+ self._lane_no * self._LANE_WIDTH))
                # if any([d < d_right and d < -0.5 for d in fp.d]):
                # if fp.d[-1] < d_right:
                if fp.d[-1] < d_right + self._lane_no * self._LANE_WIDTH:
                    rejectFlag[idx] = True
            
            elif self.both_invasion:
                # if not self._current_waypoint.is_junction:
                    # print("is junction")            ##  HDMAP FOR JUNCTION GIVES WRONG LANEMARKING FOR RIGHT/LEFT LANE
                # else:
                    # d = [i for i in fp.d]
                # print("d {}, d_right {}, d_left {}".format(fp.d[-1], d_right+ self._lane_no * self._LANE_WIDTH, d_left+ self._lane_no * self._LANE_WIDTH))
                    # if any([d < d_right and d < -0.5 for d in fp.d]):
                if fp.d[-1] < d_right + self._lane_no * self._LANE_WIDTH:
                    rejectFlag[idx] = True

                elif fp.d[-1] > d_left + self._lane_no * self._LANE_WIDTH:
                    rejectFlag[idx] = True

            for i in range(len(fp.x)-1):
                p = Point()
                p.x, p.y, p.z = fp.x[i+1], fp.y[i+1], 0
                msg.points.append(p)
                
                if self.check_ob_collision_path(p):
                    #   print("ABOUT TO COLLIDE")
                        rejectFlag[idx] = True
                    #   self._target_speed = 30
                    #   self._sf_d = 30
            if rejectFlag[idx]:
                msg.color.r, msg.color.g, msg.color.b, msg.color.a = 128.0, 0.0, 0.0, 2.0
                rejectPaths.markers.append(msg)
            else:
                acceptPaths.markers.append(msg)  
        
        self._acceptPaths_pub.publish(acceptPaths)
        self._rejectPaths_pub.publish(rejectPaths)

        if abs(self._si - self._maps[-2]) < 3:
            print("3m to goal! STOPPING!")
            self._target_speed = 0
            
        if all(rejectFlag):
            print("STOP!!")
            self._target_speed = 0
        min_cost = float("inf")
        best_path = fplist[len(fplist)/2]
        for i, fp in enumerate(fplist):
            if rejectFlag[i]:
                continue
            cost = fp.c_tot
            if min_cost >= cost:
                min_cost = cost
                best_path = fp

        
        # print("best_path",best_path.x)
        # target_s = -1000
        if self._vehicle_following:
            self._x = best_path.x[int(len(best_path.x))-1]
            self._y = best_path.y[int(len(best_path.x))-1]
            # target_s = best_path.s[int(len(best_path.x)/1.5)]

        
        else:            
            self._x = best_path.x[int(len(best_path.x)/1.5)]
            self._y = best_path.y[int(len(best_path.x)/1.5)]        
            # target_s = best_path.s[int(len(best_path.x)/1.5)]

        chosenPath = MarkerArray()
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.get_rostime()
        msg.pose.orientation.w = 1.0
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD
        msg.color.r, msg.color.g, msg.color.b, msg.color.a = 128.0, 128.0, 0, 5.0
        msg.scale.x, msg.scale.y, msg.scale.z = 0.2, 0.2, 0.2
        for count in range(len(best_path.x)-1):
            p = Point()
            p.x, p.y, p.z = best_path.x[count+1], best_path.y[count+1], 0
            msg.points.append(p)
        
        self._ref_path_pub.publish(msg)
        chosenPath.markers.append(msg)

        self._chosenPath_publisher.publish(chosenPath)
        self._target_speed_pub.publish(self._target_speed)
        # # target waypoint
        # self.target_route_point = Pose()
        # target_point = PointStamped()
        # target_point.header.frame_id = "map"
        # target_point.point.x = self._x
        # target_point.point.y = self._y
        # target_point.point.z = 0

        # # target_point.point.z = self.target_route_point.position.z
        # # self.target_route_point
        # self.target_route_point.position.x = self._x
        # self.target_route_point.position.y = self._y
        # # print("self.target_route_point",self.target_route_point)
        # self._target_point_publisher.publish(target_point)
        # # print("self.target_route_point ",self.target_route_point)
        # # self._next_waypoint = self.get_waypoint(self.target_route_point.position)
        # if self._next_waypoint.is_junction:
        #     # print("JUNCTION UP AHEAD. Slowing down")
        #     self._target_speed = 20
        #     self._sf_d = 20
        # else:
            # self._next_waypoint
        
        # # print("s {}, self._maps {}".format(s, self._maps[-1]))
        # if self._stop or all(rejectFlag):
        #     # print("STOP, or No path")
        #     control = self._vehicle_controller.run_step(
        #         0, current_speed, current_pose, self.target_route_point)
        #     control.brake = 1.0
        # elif best_path.s[-1] >= self._maps[-2]:
        #     # print("END OF ROUTE")
        #     control = self._vehicle_controller.run_step(
        #     0, current_speed, current_pose, self.target_route_point)
        #     control.brake = 1.0
        #     # self._finished = True
        #     return control, True


        # else:
        #     # print("Follow target")
        #     control = self._vehicle_controller.run_step(
        #     self._target_speed, current_speed, current_pose, self.target_route_point)
        #     # if self._target_speed < 0.5:
        #     #     control.brake = 1.0

        # print("self._target_speed {}, current_speed {}".format(self._target_speed, current_speed))
    
        # end = time.time()

        # print("final target s: {}, goal_s {}".format(best_path.s[-1], self._maps[-1]))
        # print("elapsed time {}, frequency {}".format(end-start, 1/(end-start)))
        # return control, False
        # print("FOT STARTED")
        # rospy.spin()        
        
if __name__ == '__main__':
    rospy.init_node('fotPlanner')
    my_local_planner = MyLocalPlanner()
    # rospy.spin() 
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()       