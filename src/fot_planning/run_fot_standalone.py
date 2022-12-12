#!/usr/bin/env python

# import rospy
import math
# import numpy as np
# from geometry_msgs.msg import PointStamped
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Point
# from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
# from nav_msgs.msg import Odometry
from obj_msgs.msg import Obj, ObjList
from geometry_msgs.msg import TransformStamped
""" import library for FOT """
# from PythonRobotics.PathPlanning.FrenetOptimalTrajectory import frenet_optimal_trajectory as FOT
import fot as FOT
# import time
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
import copy
from re import search
import numpy as np
import rospy
from path_msgs.msg import Trajectory
from path_msgs.msg import Map, Lane, Link, Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from control_msgs.msg import VehicleState
import time
import sys
import bisect

# from obj_msgs.msg import Obj
import tf
from math import sqrt, pow
# import math
# from spline import *
# from frenet_trajectory import *



###### from spline.py
class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        #print("dx", dx)
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        #print("result: ", result)
        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


class MySpline2D:
    def __init__(self, wx, wy, ws):
        self.s = ws
        # self.sx = wx
        # self.sy = wy
        self.sx = Spline(self.s, wx)
        self.sy = Spline(self.s, wy)

    def calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw
###### from spline.py


class my_local_planner():
    
    def __init__(self):
        self.target_route_point = None
        # self._current_waypoint = None
        # self._vehicle_controller = None
        # self._waypoints_queue = deque(maxlen=20000)
        # self._buffer_size = 5
        # self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._vehicle_yaw = 0.0
        # self._current_speed = None
        # self._current_pose = None
        # self._current_acceleration = None
        self._obstacles = []
        # self._obstacles_bbox_center = []
        # self._wp_follower = False
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
        # self._obs = []
        # self._target_speed = 80
        self._vehicle_following = False
        self._no_paths = False
        # self._junction_encountered_before = False
        # self._next_waypoint = False
        self._MAX_T = 1.4
        self._MIN_T = 1.3
        self._stop = False
        # self._ego_vehicle = None
        self._lane_change =None
        self._obj_list = []

        # self.target_route_point = None
        self._current_pose = Odometry()
        # self.frenet_hyper = FrenetHyper()
        self.global_trajectory_ = []
        self.best_path = None
        # self.fplist = None
        self._current_speed = 0.0
        self._current_acceleration = 0.0
        # self.ref_path = None
        self.prev_index_ = -1
        self.ego_ = Obj()

        # self.PathArray = Marker()
        # self.init_global()

        self.opt_path_idx = -1

        self.sx_ = None
        self.sy_ = None
        self.s_  = None

        self.offsetX_ = 0.0
        self.offsetY_ = 0.0

        self.links_ = []
        self.lanes_ = []
        self.nodes_ = []

        self.OType = { 
            "NOINTEREST" : 0, 
            "APPROACHING" : 1, 
            "STATIC" : 2,
            "FRONTOBJ"  : 3, 
            "OPPOSITE" : 4,
            "RED" : 5,
            "EGO" : 6,
            "GOAL" : 7,
            "STOP" : 8,
            "ATYPICAL" : 9,
            "LANES" : 10,
            "FAKE" : 11
            }
        
        self.LCode = {
            "CENTER" : 1,
            "UTURN" : 2,
            "LANE" : 3,
            "NO_LANE_CHANGE" : 5,
            "NO_PARKING" : 8,
            "NO_STOPPING" : 9,
            "BARRIER" : 99
        }

        #### subscribers ####
        self.global_trajectory_sub_ = rospy.Subscriber("/global_trajectory",  Trajectory, self.trajectoryCallback)
        self.current_odom_sub_ = rospy.Subscriber("/gps_odom",  Odometry, self.currentOdomCallback, queue_size=1)
        self.obj_info_sub_ = rospy.Subscriber("/obj_info",  ObjList, self.objCallback, queue_size=1)
        self.map_info_sub_ = rospy.Subscriber("/map_info", Map, self.mapInfoCallback)
        self.ego_state_sub_ = rospy.Subscriber("/vehicle_state", VehicleState, self.egoStateCallback)
        
        self._acceptPaths_pub = rospy.Publisher("/accept_paths", MarkerArray, queue_size=1)
        self._rejectPaths_pub = rospy.Publisher("/reject_paths", MarkerArray, queue_size=1)
        self._chosenPath_publisher = rospy.Publisher("/chosen_path", MarkerArray, queue_size=1)
        self._ref_path_pub = rospy.Publisher("/ref_path", Marker, queue_size=1)
        self._target_speed_pub = rospy.Publisher("/Target_Velocity", Float64, queue_size=1)


        #### flag list ###
        self.pose_initialized_ = False
        self.trajectory_initialized_ = False
        self.map_initialized_ = False

        ### current not using
        self.closest_global_way_index = 0
        self._current_glob_idx = 0

        ### map ###

    #### callback function ####
    # def init_global(self):
    #     self.PathArray.header.frame_id = "map"
    #     self.PathArray.header.stamp = rospy.get_rostime()
    #     self.PathArray.action = Marker.ADD
    #     self.PathArray.pose.orientation.w = 1.0
    #     self.PathArray.id = 335
    #     self.PathArray.type = Marker.LINE_STRIP
    #     self.PathArray.scale.x = 0.8
    #     self.PathArray.color.g = 0.5
    #     self.PathArray.color.b = 20.0
    #     self.PathArray.color.a = 1.0
    def transformObj(self,src):
        base_link2map = tf.Transformer()
        m = TransformStamped()
        m.header.frame_id = "/map"
        m.child_frame_id = "/base_link"
        m.transform.translation = self.ego_.pose.position
        m.transform.rotation = self.ego_.pose.orientation
        print("self.ego_.pose.orientation ",self.ego_.pose.orientation)
        base_link2map.setTransform(m)
        # print("base_link2map: ",base_link2map)

        base_link2obj = tf.Transformer()
        l = TransformStamped()
        print("src.pose.position, ",src.pose.position)
        print("src.pose.orientation, ",src.pose.orientation)
        l.header.frame_id = "/base_link"
        l.child_frame_id = "/obj"
        l.transform.translation = src.pose.position
        l.transform.rotation.w = 1
        # l.transform.rotation = src.pose.orientation
        base_link2obj.setTransform(l)
        # print("base_link2obj: ",base_link2obj)
        # listener = tf.TransformListener()
        bwrtmap_trans, bwrtmap_ori = base_link2map.lookupTransform("/map", "/base_link", rospy.Time(0))
        trans1_mat = tf.transformations.translation_matrix(bwrtmap_trans)
        trans1_rot = tf.transformations.quaternion_matrix(bwrtmap_ori)
        mat1 = np.dot(trans1_mat,trans1_rot)
        srcwrtb_trans, srcwrtb_ori = base_link2obj.lookupTransform("/base_link", "/obj", rospy.Time(0))
        trans2_mat = tf.transformations.translation_matrix(srcwrtb_trans)
        trans2_rot = tf.transformations.quaternion_matrix(srcwrtb_ori)
        mat2 = np.dot(trans2_mat,trans2_rot)
        mat3 = np.dot(mat1,mat2)
        print("mat3", mat3)
        gpose = tf.transformations.translation_from_matrix(mat3)
        print("obs pose: ",gpose[0], gpose[1])

        print(gpose)
        # map2obj = base_link2map * base_link2obj
        print(map2obj)
        



    def objCallback(self, obj_msg):
        self._obj_list = []
        for i in range(len(obj_msg.objlist)-1):
            lobj = obj_msg.objlist[i]
            # print("lobj",lobj)
            self.transformObj(lobj)

    def trajectoryCallback(self, traj_msg):
        print("TRAJ CALLBACK")
        if len(traj_msg.waypoints) > 0:
            self.global_trajectory_ = []
            for i in range(len(traj_msg.waypoints)):
                # if(i % 2 == 0):
                self.global_trajectory_.append(traj_msg.waypoints[i])
                # self.PathArray.points.append(traj_msg.waypoints[i].point)

            self.prev_index_ = -1
            self.generateCenterLine()
            
            self.trajectory_initialized_ = True
            
        else:
            self.trajectory_initialized_ = False

    def generateCenterLine(self):
        length = 0.0
        wx = [self.global_trajectory_[0].point.x]
        wy = [self.global_trajectory_[0].point.y]
        ws = [length]

        for i in range(1, len(self.global_trajectory_)):
            wp = self.global_trajectory_[i].point
            prev_wp = self.global_trajectory_[i-1].point

            step = sqrt(pow(prev_wp.x - wp.x, 2) + pow(prev_wp.y - wp.y, 2))
            if step > 0.0:
                length += step
                wx.append(self.global_trajectory_[i].point.x)
                wy.append(self.global_trajectory_[i].point.y)
                ws.append(length)

        self.ref_path = MySpline2D(wx, wy, ws)
        
        self._mapx = np.zeros(len(wx)-1)
        for i in range(len(wx)-1):
            self._mapx[i] = wx[i]
        
        self._mapy = np.zeros(len(wy)-1)
        for i in range(len(wy)-1):
            self._mapy[i] = wy[i]
        
        self._maps = np.zeros(len(ws)-1)
        for i in range(len(ws)-1):
            self._maps[i] = ws[i]
        
        self.sx_ = self.ref_path.sx
        self.sy_ = self.ref_path.sy
        self.global_len_ = length
        self.s_ = ws


    def egoStateCallback(self, vehicle_state_msg):
        self._current_speed = vehicle_state_msg.v_ego
        self._current_acceleration = vehicle_state_msg.a_x
        # print("current_speed",self._current_acceleration)


    def currentOdomCallback(self, odom_msg):
        # print("ODOM CALLBACK")
        time1 = time.time()
            # self._current_pose.header.stamp = odom_msg.header.stamp
            # self._current_pose.header.frame_id = "map"
            # self._current_pose.pose = odom_msg.pose.pose

            # self.ego_.id = 0
            # self.ego_.type = self.OType["EGO"]
            # print(self.ego_.type)
        self.ego_.pose = odom_msg.pose.pose
        # print("Current pose:",self.ego_.pose.position.x, self.ego_.pose.position.y)
        # self.ego_.pose = odom_msg.pose.pose.orientation
        # print(self.ego_)
        quaternion = (
            self.ego_.pose.orientation.x,
            self.ego_.pose.orientation.y,
            self.ego_.pose.orientation.z,
            self.ego_.pose.orientation.w
        )
        _, _, self._vehicle_yaw = euler_from_quaternion(quaternion)
    
        self.pose_initialized_ = True
        if(self.map_initialized_):
            self.startThread()
        time2 = time.time()
        # print("freq",(1/(time2-time1)))

    def startThread(self):
        start_time_1 = time.time()

        if self.trajectory_initialized_ and self.pose_initialized_:
            self.locateEgo()
            # self.fplist = calc_frenet_paths_total(self.frenet_hyper, self.current_speed / 3.6, self.ego_.lat_offset, 0.0, 0.0, self.ego_.s, 35.0 / 3.6, self.ref_path)
            clink = None
            for link in self.links_:
                if link[0] == self.ego_.link_id:
                    clink = link
                    break
            self.laneStatus_2(clink)
            self.run_step()
            # self.findBestPath()
            # # self.PathArrayPub.publish(self.PathArray)
            # self.visualize_frenet_path(self.opt_path_idx)
            

        # print("total time: ", (1/(time.time() - start_time_1)))
    
    def findBestPath(self):
        min_cost = float("inf")
        for i, fp in enumerate(self.fplist):
            if fp.allowLine:
                if min_cost >= fp.cf:
                    min_cost = fp.cf
                    self.opt_path = fp
                    self.opt_path_idx = i

    def locateEgo(self):
        point = Point()
        point.x = self.ego_.pose.position.x
        point.y = self.ego_.pose.position.y 
        min_idx = 0
        if self.prev_index_ != -1:
            min_idx = self.prev_index_
        min_dist = 9999
        if min_idx == 0:
            for i in range(min_idx, len(self.global_trajectory_)):
                tmp_dist = sqrt(pow(point.x - self.global_trajectory_[i].point.x, 2) + pow(point.y - self.global_trajectory_[i].point.y, 2))
                if (tmp_dist < min_dist):
                    min_dist = tmp_dist
                    min_idx = i

        elif min_idx > 0:
            for i in range(min_idx, min_idx + 10):
                tmp_dist = sqrt(pow(point.x - self.global_trajectory_[i].point.x, 2) + pow(point.y - self.global_trajectory_[i].point.y, 2))
                if (tmp_dist < min_dist):
                    min_dist = tmp_dist
                    min_idx = i

        self.ego_.index = min_idx
        self.prev_index_ = self.ego_.index
        self.ego_.s = self.s_[self.ego_.index]
        self.ego_.lat_offset = self.getD()
        
        self.ego_.link_id = self.getLinkID(self.ego_) 
    def getLinkID(self, obj):
        position = obj.pose.position

        link_id = self.global_trajectory_[obj.index].CLID
        # print("****link id: ******", link_id)
        clink = None
        for link in self.links_:
            if link[0] == link_id:
                clink = link
                break
        # print("clink1_center:", link_id)
        # tmp_txt1 = link_id
        clink_idx = self.searchLinkIndex(position, clink)
        # print("clink_idx",clink_idx)
        # print("ego_x ",position.x ," ego_y ",position.y )
        # print(" clink_x, ",clink[-1].geometry[clink_idx].x - self.offsetX_, " clink_y, ",clink[-1].geometry[clink_idx].y-self.offsetY_)
        cdist = sqrt(pow(position.x - (clink[-1].geometry[clink_idx].x - self.offsetX_), 2) + pow(position.y - (clink[-1].geometry[clink_idx].y - self.offsetY_), 2))
        # print("cdist {}".format(cdist))


        if len(clink[-1].RLID[0]) != 0:
            rlink = None
            for link in self.links_:
                if link[0] == clink[-1].RLID[0]:
                    rlink = link
                    break
            rlink_idx = self.searchLinkIndex(position, rlink)
            rdist = sqrt(pow(position.x - (rlink[-1].geometry[rlink_idx].x - self.offsetX_), 2) + pow(position.y - (rlink[-1].geometry[rlink_idx].y - self.offsetY_),2))
            dist = sqrt(pow(rlink[-1].geometry[rlink_idx].x - clink[-1].geometry[clink_idx].x, 2) + pow(rlink[-1].geometry[rlink_idx].y - clink[-1].geometry[clink_idx].y, 2))
            # print("distance between clink and rlink: ",dist)
            # print(" rlink_x, ",rlink[-1].geometry[rlink_idx].x - self.offsetX_, " rlink_y, ",rlink[-1].geometry[rlink_idx].y - self.offsetY_)

            # print("rdist",rdist)

            if rdist < cdist:
                link_id = clink[-1].RLID[0]
                cdist = rdist

                # print("ego closer to RLINK, linkid_updated ",link_id )

        if len(clink[-1].LLID[0]) != 0:
            llink = None
            for link in self.links_:
                if link[0] == clink[-1].LLID[0]:
                    llink = link
                    break
            llink_idx = self.searchLinkIndex(position, llink)
            ldist = sqrt(pow(position.x - (llink[-1].geometry[llink_idx].x - self.offsetX_), 2) + pow(position.y - (llink[-1].geometry[llink_idx].y - self.offsetY_),2))
            # print("ldist",ldist)
            dist = sqrt(pow(llink[-1].geometry[llink_idx].x - clink[-1].geometry[clink_idx].x, 2) + pow(llink[-1].geometry[llink_idx].y - clink[-1].geometry[clink_idx].y, 2))
            # print("distance between clink and llink: ",dist)
            # print(" llink_x, ",llink[-1].geometry[llink_idx].x - self.offsetX_, " llink_y, ",llink[-1].geometry[llink_idx].y - self.offsetY_)

            if ldist < cdist:
                link_id = clink[-1].LLID[0]
                cdist = ldist
                # print("ego closer to LLINK, linkid_updated ",link_id )


        # print("clink2:", link_id)

        return link_id


    def laneStatus_2(self, clink):
        # print("R_laneID: ", len(clink[-1].R_laneID))
        # print("L_laneID: ", len(clink[-1].L_laneID))

        # if clink[-1].LLID[0] == "" and clink[-1].RLID[0] == "":
        # if len(clink[-1].LLID) == 0 and len(clink[-1].RLID) == 0:
            # for fp in self.fplist:
            #     if fp.wayType == 0:
            #         fp.allowLine = True
        # print("clink",clink[-1].id)
        # print("clink[-1].LLID[0]",clink[-1].LLID[0])
        # print("clink[-1].RLID[0]",clink[-1].RLID[0])

        if clink[-1].LLID[0] == "" and clink[-1].RLID[0] == "":
            self._lane_change = "KEEP"

        elif clink[-1].LLID[0] != "" and clink[-1].RLID[0] != "":
            self._lane_change = "BOTH"

        elif clink[-1].LLID[0] != "" and clink[-1].RLID[0] == "":
            self._lane_change = "LEFT"

        elif clink[-1].LLID[0] == "" and clink[-1].RLID[0] != "":
            self._lane_change = "RIGHT"
        # elif len(clink[-1].LLID)!= 0 and len(clink[-1].RLID) == 0:

        # print("Lane_status: ",self._lane_change)

        #     for fp in self.fplist:
        #         if fp.wayType >= 0:
        #             fp.allowLine = True
        # elif clink[-1].LLID[0] == "" and clink[-1].RLID[0] != "":
        #     for fp in self.fplist:
        #         if fp.wayType <= 0:
        #             fp.allowLine = True
        # else:
        #     for fp in self.fplist:
        #         fp.allowLine = True

        # if clink[-1].R_laneID == self.LCode["CENTER"]:
        #     for fp in self.fplist:
        #         if fp.wayType < 0:
        #             fp.allowLine = False

        # elif clink[-1].L_laneID == self.LCode["CENTER"]:
        #     for fp in self.fplist:
        #         if fp.wayType > 0:
        #             fp.allowLine = False


        

    def searchLinkIndex(self, point, link):
        min_idx = 0
        min_dist = sqrt(pow(point.x - link[-1].geometry[0].x - self.offsetX_, 2) + pow(point.y - link[-1].geometry[0].y - self.offsetY_, 2))

        for i in range(1, len(link[-1].geometry)):
            dist = sqrt(pow(point.x - link[-1].geometry[i].x - self.offsetX_, 2) + pow(point.y - link[-1].geometry[i].y - self.offsetY_, 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx
    
    def getD(self):

        prev = self.global_trajectory_[self.ego_.index].point
        next = self.global_trajectory_[self.ego_.index + 1].point

        cx = self.ego_.pose.position.x
        cy = self.ego_.pose.position.y
        lat_offset = sqrt(pow(cx - prev.x, 2) + pow(cy - prev.y, 2))

        ego_vec = [cx - prev.x, cy - prev.y, 0]
        map_vec = [next.x - prev.x, next.y - prev.y, 0]
        d_cross = np.cross(ego_vec,map_vec)
        if d_cross[-1] > 0:
            lat_offset = -lat_offset
        
        return lat_offset

    def mapInfoCallback(self, map_msg):
        map_info = map_msg

        for i in range(len(map_info.links)):
            link = map_info.links[i]
            self.links_.append((link.id, link))
        
        for i in range(len(map_info.nodes)):
            node = map_info.nodes[i]
            self.nodes_.append((node.id, node))

     
        for i in range(len(map_info.lanes)):
            lane = map_info.lanes[i]
            self.lanes_.append((lane.id, lane))

        self.offsetX_ = map_info.OffsetMapX
        self.offsetY_ = map_info.OffsetMapY

        self.map_initialized_ = True
        
    def distanceToIndex(self, csp, idx, dist):
        s_start = csp.s[idx]

        for i in range(idx, len(csp.s)):
            if(csp.s[i] > s_start + dist):
                return i
        return len(csp.s) - 1

    def check_ob_collision_path(self, point):   # , l_vector, r_vector
        point.z = point.z + 0.75
        # return
        # print("POINT: \n", point)
        # print(self._obj_list)
        for ob in self._obj_list:

            # if self.check_obstacle(point, ob):
            if self.fot_collision_check(point, ob):

                return True
        return False



    def fot_collision_check(self, point, obstacle):
        # center = [obstacle.ros_transform.position.x, obstacle.ros_transform.position.y]
        obs_center = obstacle
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
        if self._obstacles:
            time1 = time.time()
            i = 0
            for ob in self._obstacles:
                i = i + 1
                # ob_s, ob_d = FOT.get_frenet(ob.ros_transform.position.x, ob.ros_transform.position.y, self._mapx, self._mapy)
                p1 = c_lib.get_frenet(c_double(ob.ros_transform.position.x), c_double(ob.ros_transform.position.y), self._mapx.ctypes.data_as(c_void_p), self._mapy.ctypes.data_as(c_void_p), c_long(len(self._mapx)))
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
                    if ((self.both_invasion or (self._no_paths and (not self._vehicle_following)) or self._current_waypoint.is_junction) and ((ob_s - s) > 0 and (ob_s - s) < 15)):         
                        # self._sf = ob_s - (D0 - T * (ob.speed/3.6))
                        self._sf = ob_s - 5

                        print(self._sf)
                        self._target_speed = ob.speed
                        self._sf_d = ob.speed
                        self._vehicle_following = True

                        # if (ob_s - s) > 0 and (ob_s - s) < 10 and (current_speed - ob.speed) > 5  :
                        if (current_speed - ob.speed) > 5  :
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
                    elif (abs(ob.speed - 30) < 10) and ((ob_s - s) > 0 (ob_s - s) < 15):        

                        # self._sf = ob_s - (D0 - T * (ob.speed/3.6))
                        self._sf = ob_s - 5
                        print(self._sf)
                        self._target_speed = ob.speed
                        self._sf_d = ob.speed
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
                        self._target_speed = 30 -   10 * abs(abs(self._vehicle_yaw) - abs(self._yaw_road))/(2*np.pi)
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
                    self._target_speed = ob.speed
                    self._sf_d = ob.speed
                    self._vehicle_following = True

                    # if (ob_s - s) > 0 and (ob_s - s) < 10 and (current_speed - ob.speed) > 5  :
                    if (current_speed - ob.speed) > 5  :
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
                    self._target_speed = 30 -   10 * abs(abs(self._vehicle_yaw) - abs(self._yaw_road))/(2*np.pi)
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
            self._target_speed = 30 -   10 * abs(abs(self._vehicle_yaw) - abs(self._yaw_road))/np.pi

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


        # x = self._current_pose.pose.position.x
        # y = self._current_pose.pose.position.y
        x = self.ego_.pose.position.x
        y = self.ego_.pose.position.y
        # print("Current pose:",x, ", ",y)
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
        print("FOT STARTED")
        # rospy.spin()        
        
            
if __name__ == '__main__':
    rospy.init_node('fotPlanner')
    my_local_planner = my_local_planner()
    # rospy.spin() 
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()     