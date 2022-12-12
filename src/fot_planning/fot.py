
#-*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import math

from numpy import *
from matplotlib import *

# import cubic_spline as csp
# with open('./map/map_coord_proto2.pkl', 'rb') as f:
#     map_coord = pickle.load(f)

# map_in = map_coord['Lane_inner']
# map_center = map_coord['Lane_center']
# map_out = map_coord['Lane_outer']
# wp_in = map_coord['waypoint_inner']
# wp_out = map_coord['waypoint_outer']
from ctypes import*
import os

""" THESE ARE RELATED TO IMPORTING C get_frenet() """
libname = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "fot_cpp_1.so"))

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


# initialize
V_MAX = 100      # maximum velocity [m/s]
ACC_MAX = 5000 # maximum acceleration [m/ss]
K_MAX = 50     # maximum curvature [1/m]

TARGET_SPEED = 20 / 3.6 # target speed [m/s]
LANE_WIDTH = 1  # lane width [m]

COL_CHECK = 4.25 # collision check distance [m]

MIN_T = 1.1 # minimum terminal time [s]
MAX_T = 1.25 # maximum terminal time [s]
DT_T = 0.2 # sampling timestep
DT = 0.1 # timestep interval between the calculated points
DT_SAMPLE = 0.4 # lane_width sampling rate
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed


# cost weights
K_J = 0.01 # weight for jerk
K_T = 0.01 # weight for terminal time
K_D = 19.5 # weight for consistency
K_V = 0.1 # weight for getting to target speed
K_LAT = 0.095 # weight for lateral direction
K_LON = 0.1 # weight for longitudinal direction

SIM_STEP = 500 # simulation step
SHOW_ANIMATION = True # plot 으로 결과 보여줄지 말지

# Vehicle parameters - plot 을 위한 파라미터
LENGTH = 0.39  # [m]
WIDTH = 0.19  # [m]
BACKTOWHEEL = 0.1  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.22  # [m]

OFFSET = 1.5 # distance to keep away from lane

# lateral planning 시 terminal position condition 후보  (양 차선 중앙)
# DF_SET = np.array([LANE_WIDTH/2, -LANE_WIDTH/2])

def even_set(a,b):
    # a is the lower number
    # b is the higher number
    a = np.floor(a)
    if a % 2 != 0:
        a = a - 1
    b = np.floor(b)
    if b % 2 != 0:
        b = b + 1
    
    DF_SET = np.arange(a,b+2,1)
    return DF_SET
    # print((DF_SET))

def next_waypoint(x, y, mapx, mapy):
    closest_wp = get_closest_waypoints(x, y, mapx, mapy)
    # print("closest_wp {}\n, mapx {}\n, mapy {}\n".format(closest_wp,mapx,mapy))
    try:
        map_vec = [mapx[closest_wp + 1] - mapx[closest_wp], mapy[closest_wp + 1] - mapy[closest_wp]]
    except IndexError:
        return False
    except:
        print("closest_wp {}\n, mapx {}\n, mapy {}\n".format(closest_wp,mapx,mapy))

    map_vec = [mapx[closest_wp + 1] - mapx[closest_wp], mapy[closest_wp + 1] - mapy[closest_wp]]
    ego_vec = [x - mapx[closest_wp], y - mapy[closest_wp]]

    direction  = np.sign(np.dot(map_vec, ego_vec))

    if direction >= 0:
        next_wp = closest_wp + 1
    else:
        next_wp = closest_wp

    # print("closest_wp {}, next_wp {}".format(closest_wp, next_wp))
    return next_wp


def get_closest_waypoints(x, y, mapx, mapy):
    min_len = 1e10
    closest_wp = 0

    for i in range(len(mapx)):
        _mapx = mapx[i]
        _mapy = mapy[i]
        dist = get_dist(x, y, _mapx, _mapy)

        if dist < min_len:
            min_len = dist
            closest_wp = i

    return closest_wp


def get_dist(x, y, _x, _y):
    return np.sqrt((x - _x)**2 + (y - _y)**2)

def get_frenet(x, y, mapx, mapy):
    next_wp = next_waypoint(x, y, mapx, mapy)
    if next_wp == False:
        return False, False

    if (next_wp - 2) > 0:

        prev_wp = next_wp - 2
    else:
        next_wp = next_wp + 2
        prev_wp = 0

    n_x = mapx[next_wp] - mapx[prev_wp]
    n_y = mapy[next_wp] - mapy[prev_wp]
    x_x = x - mapx[prev_wp]
    x_y = y - mapy[prev_wp]

    proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y)
    proj_x = proj_norm*n_x
    proj_y = proj_norm*n_y

    # print ("proj_x {}, proj_y {}".format(proj_x, proj_y))
    #-------- get frenet d
    frenet_d = get_dist(x_x,x_y,proj_x,proj_y)

    ego_vec = [x-mapx[prev_wp], y-mapy[prev_wp], 0];
    map_vec = [n_x, n_y, 0];
    d_cross = np.cross(ego_vec,map_vec)
    # print("d_cross ", d_cross)
    if d_cross[-1] > 0:
        frenet_d = -frenet_d;

    #-------- get frenet s
    frenet_s = 0
    for i in range(prev_wp):
        frenet_s = frenet_s + get_dist(mapx[i],mapy[i],mapx[i+1],mapy[i+1]);

    frenet_s = frenet_s + get_dist(0,0,proj_x,proj_y);

    return frenet_s, frenet_d


# def get_cartesian(s, d, mapx, mapy, maps):
#     prev_wp = 0
#     # print("mapx ", mapx)
#     # print("mapy ", mapy)
#     # print("maps ", maps)
#     # print("S: ",s)
#     s = np.mod(s, maps[-2])
#     # print("s {}\n, maps[prew_wp+1] {}\n ,s > maps[prev_wp+1]\n ".format(s, maps[prev_wp+1],s > maps[prev_wp+1]))
#     # print("prev_wp < len(maps)-2", prev_wp < len(maps)-2)
#     # print("s {}, maps[prev_wp+1] {}, prev_wp {} < len(maps)-2 {})".format(s, maps[prev_wp+1], prev_wp, len(maps)-2))
#     while(np.all((s > maps[prev_wp+1])) and ((prev_wp < len(maps)-2))):
#         prev_wp = prev_wp + 1

#     next_wp = np.mod(prev_wp+1,len(mapx))

#     dx = (mapx[next_wp]-mapx[prev_wp])
#     dy = (mapy[next_wp]-mapy[prev_wp])

#     heading = np.arctan2(dy, dx) # [rad]

#     # the x,y,s along the segment
#     seg_s = s - maps[prev_wp];

#     seg_x = mapx[prev_wp] + seg_s*np.cos(heading);
#     seg_y = mapy[prev_wp] + seg_s*np.sin(heading);

#     perp_heading = heading + 90 * np.pi/180;
#     x = seg_x + d*np.cos(perp_heading);
#     y = seg_y + d*np.sin(perp_heading);

#     return x, y, heading


def get_cartesian(s,d,mapx, mapy, wp_s):
    prev_wp = 0
    # print("wp_s ", wp_s)
    # print("INSIDE FOT: s ",s, " wp_s[-1] ",wp_s[-1])
    s = np.mod(s, wp_s[-1]) # EDITED
    while (s > wp_s[prev_wp + 1]) and (prev_wp < len(wp_s) - 2):
        prev_wp = prev_wp + 1

    next_wp = np.mod(prev_wp + 1, len(mapx))
    dx = (mapx[next_wp] - mapx[prev_wp])
    dy = (mapy[next_wp] - mapy[prev_wp])

    heading = np.arctan2(dy, dx)

    seg_s = s - wp_s[prev_wp];

    seg_x = mapx[prev_wp] + seg_s*np.cos(heading);
    seg_y = mapy[prev_wp] + seg_s*np.sin(heading);

    perp_heading = heading + 90 * np.pi/180;
    x = seg_x + d*np.cos(perp_heading);
    y = seg_y + d*np.sin(perp_heading);

    return x,y,heading


class QuinticPolynomial:

    def __init__(self, xi, vi, ai, xf, vf, af, T):
        # calculate coefficient of quintic polynomial
        # used for lateral trajectory
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T** 4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([xf - self.a0 - self.a1*T - self.a2*T**2,
                      vf - self.a1 - 2*self.a2*T,
                      af - 2*self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    # calculate postition info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5 * t ** 5
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
        return j

class QuarticPolynomial:

    def __init__(self, xi, vi, ai, vf, af, T):
        # calculate coefficient of quartic polynomial
        # used for longitudinal trajectory
        self.a0 = xi
        self.a1 = vi
        self.a2 = 0.5*ai

        A = np.array([[3*T**2, 4*T**3],
                             [6*T, 12*T**2]])
        b = np.array([vf - self.a1 - 2*self.a2*T,
                             af - 2*self.a2])

        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    # calculate postition info.
    def calc_pos(self, t):
        x = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4
        return x

    # calculate velocity info.
    def calc_vel(self, t):
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3
        return v

    # calculate acceleration info.
    def calc_acc(self, t):
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2
        return a

    # calculate jerk info.
    def calc_jerk(self, t):
        j = 6*self.a3 + 24*self.a4*t
        return j

class FrenetPath:

    def __init__(self):
        # time
        self.t = []

        # lateral traj in Frenet frame
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []

        # longitudinal traj in Frenet frame
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []

        # cost
        self.c_lat = 0.0
        self.c_lon = 0.0
        self.c_tot = 0.0
        self.c_offset = 0.0
        # combined traj in global frame
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.kappa = []

def calc_frenet_paths(si, si_d, si_dd, sf, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, LANE_WIDTH, Vehicle_Following, MAX_T, MIN_T):
    # print("si {}, si_d {}, si_dd {}, sf_d {}, sf_dd {}, di {}, di_d {}, di_dd {}, df_d {}, df_dd {}".format(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd))
    
    frenet_paths = []
    # print("si ",si)
    # generate path to each offset goal
    # print("dfset",DF_SET)
    # for df in np.arange(-LANE_WIDTH/2 - OFFSET,LANE_WIDTH/2 + OFFSET,DT_SAMPLE):
    if Vehicle_Following:
        # DF_SET = np.arange(-2,4,1)
        DF_SET = even_set(di - LANE_WIDTH/2, di + LANE_WIDTH/2)
        # DF_SET = np.arange(int(di - 1.15*LANE_WIDTH/2), int(di + 1.15*LANE_WIDTH/2),1)

    else:
        # DF_SET = np.arange(-6,8,2)
        DF_SET = even_set(di - LANE_WIDTH, di + LANE_WIDTH)

        # DF_SET = np.arange(int(di - 1.15*LANE_WIDTH), int(di + 1.15*LANE_WIDTH),1)
        # print("DF_SET: ",DF_SET)


    for df in DF_SET:
    # for df in np.arange(-6, 8, 2):
    #     MIN_T = MIN_T
    #     MAX_T = MAX_T
    #     # print("HELLO")
    # # for df in DF_SET:
    #     # print("df: ",df)
    #     # Lateral motion planning
    #     if Vehicle_Following :
    #         MIN_T = 3.0
    #         MAX_T = 3.1
        # print("MIN_T {}, MAX_T {}".format(MIN_T, MAX_T))
        for T in np.arange(MIN_T, MAX_T, DT_T):
            fp = FrenetPath()
            lat_traj = QuinticPolynomial(di, di_d, di_dd, df, df_d, df_dd, T)

            fp.t = [t for t in np.arange(0.0, T, DT)]
            # print("fp.t ", fp.t)
            fp.d = [lat_traj.calc_pos(t) for t in fp.t]
            fp.d_d = [lat_traj.calc_vel(t) for t in fp.t]
            fp.d_dd = [lat_traj.calc_acc(t) for t in fp.t]
            fp.d_ddd = [lat_traj.calc_jerk(t) for t in fp.t]

            # Longitudinal motion planning (velocity keeping)
            
        # for tv in np.arange(sf_d/3.6 - D_T_S * N_S_SAMPLE,
        #             sf_d/3.6 + D_T_S * N_S_SAMPLE, D_T_S):
                
            tfp = deepcopy(fp)
            
            if Vehicle_Following:
                # print("si {}, si_d {}, si_dd {}, sf {}, sf_d {}, sf_dd {}, T {}".format(si, si_d, si_dd, sf, sf_d, sf_dd, T))
                lon_traj = QuinticPolynomial(si, si_d, si_dd, sf, sf_d, sf_dd, T)
            else:
                # print("si {}, si_d {}, si_dd {}, sf_d {}, sf_dd {}, T {}".format(si, si_d, si_dd, sf_d, sf_dd, T))

                lon_traj = QuarticPolynomial(si, si_d, si_dd, sf_d, sf_dd, T)
            # print("lon_traj", lon_traj)
            # print("lon_traj.calc_pos(0.1)", lon_traj.calc_pos(0.1))
            
            tfp.s = [lon_traj.calc_pos(t) for t in fp.t]
            # print("tfp.s ", tfp.s)
            tfp.s_d = [lon_traj.calc_vel(t) for t in fp.t]
            tfp.s_dd = [lon_traj.calc_acc(t) for t in fp.t]
            tfp.s_ddd = [lon_traj.calc_jerk(t) for t in fp.t]


            # # 경로 늘려주기 (In case T < MAX_T)
            # for _t in np.arange(T, MAX_T, DT):
            #     tfp.t.append(_t)
            #     tfp.d.append(tfp.d[-1])
            #     _s = tfp.s[-1] + tfp.s_d[-1] * DT
            #     tfp.s.append(_s)

            #     tfp.s_d.append(tfp.s_d[-1])
            #     tfp.s_dd.append(tfp.s_dd[-1])
            #     tfp.s_ddd.append(tfp.s_ddd[-1])

            #     tfp.d_d.append(tfp.d_d[-1])
            #     tfp.d_dd.append(tfp.d_dd[-1])
            #     tfp.d_ddd.append(tfp.d_ddd[-1])

            J_lat = sum(np.power(tfp.d_ddd, 2))  # lateral jerk
            J_lon = sum(np.power(tfp.s_ddd, 2))  # longitudinal jerk

            # cost for consistency
            # d_diff = (tfp.d[-1] - opt_d) ** 2
            # cost for target speed
            v_diff = (TARGET_SPEED - tfp.s_d[-1]) ** 2

            # offset_cost = sum(np.power(tfp.d, 2))
            # lateral cost
            # tfp.c_lat = K_J * J_lat + K_T * T + K_D * offset_cost ** 2

            tfp.c_offset = tfp.d[-1] ** 2
            tfp.c_lat = K_J * J_lat + K_T * T + K_D * tfp.d[-1] ** 2
            # logitudinal cost
            tfp.c_lon = K_J * J_lon + K_T * T + K_V * v_diff

            # total cost combined
            tfp.c_tot = K_LAT * tfp.c_lat + K_LON * tfp.c_lon

            # print("COST {}, lateral cost {}, long. cost {}, offset cost {}".format(tfp.c_tot, tfp.c_lat, tfp.c_lon,K_D * tfp.d[-1] ** 2 ))

            frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, mapx, mapy, maps):

    # transform trajectory from Frenet to Global
    for fp in fplist:
        # print("fp.s ",fp.s)
        for i in range(len(fp.s)):
            _s = fp.s[i]
            _d = fp.d[i]
            """ for some reason _s and _d are and array of 3 elems.
                check why """
            # print("_s {}, _d {}".format(fp.s[i], fp.d[i]))
            # _x, _y, _ = get_cartesian(_s, _d, mapx, mapy, maps)
            if _s > maps[-1]:
                _s = maps[-2]   
            q1 = c_lib.get_cartesian(c_double(_s), c_double(_d), mapx.ctypes.data_as(c_void_p), mapy.ctypes.data_as(c_void_p), maps.ctypes.data_as(c_void_p), c_long(len(maps)))
            _x, _y, _ = q1.x, q1.y, q1.heading

            fp.x.append(_x)
            fp.y.append(_y)

        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(np.arctan2(dy, dx))
            fp.ds.append(np.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            yaw_diff = fp.yaw[i + 1] - fp.yaw[i]
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            fp.kappa.append(yaw_diff / fp.ds[i])

    return fplist


# def calc_global_paths(fplist, mapx, mapy, maps):

#     # transform trajectory from Frenet to Global
#     for fp in fplist:
#         # print("fp.s ",fp.s)
#         for i in range(len(fp.s)):
#             _s = fp.s[i]
#             _d = fp.d[i]
#             """ for some reason _s and _d are and array of 3 elems.
#                 check why """
#             # print("_s {}, _d {}".format(fp.s[i], fp.d[i]))
            
#             _x, _y, _ = get_cartesian(_s, _d, mapx, mapy, maps)
#             fp.x.append(_x)
#             fp.y.append(_y)

#             if i == 0:
#                 continue

#             dx = fp.x[i] - fp.x[i - 1]
#             dy = fp.y[i] - fp.y[i - 1]
#             fp.yaw.append(np.arctan2(dy, dx))
#             fp.ds.append(np.hypot(dx, dy))

#             # print("current i {} len fp.yaw {}".format(i,len(fp.yaw)))
#             if i == (len(fp.s) - 1):            
#                 fp.yaw.append(fp.yaw[-1])
#                 fp.ds.append(fp.ds[-1])
#             if len(fp.yaw) < 2:
#                 continue
#             yaw_diff = fp.yaw[i-1] - fp.yaw[i-2]
#             yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
#             fp.kappa.append(yaw_diff / fp.ds[i-1])

#         # calc curvature
#         # for i in range(len(fp.yaw) - 1):
#         #     yaw_diff = fp.yaw[i + 1] - fp.yaw[i]
#         #     yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
#         #     fp.kappa.append(yaw_diff / fp.ds[i])

#     return fplist



def collision_check(fp, obs, mapx, mapy, maps):
    for i in range(len(obs[:, 0])):
        # get obstacle's position (x,y)
        obs_xy = get_cartesian( obs[i, 0], obs[i, 1], mapx, mapy, maps)
        # print("obs_xy", obs_xy)
        d = [((_x - obs_xy[0]) ** 2 + (_y - obs_xy[1]) ** 2)
             for (_x, _y) in zip(fp.x, fp.y)]
        # print("fp.x ", fp.x, " fp.y ", fp.y)
        # print("d",d)
        collision = any([di <= COL_CHECK ** 2 for di in d])

        if collision:
            print("rejected due to collision")
            return True

    return False


def check_path(fplist, obs, mapx, mapy, maps):
    ok_ind = []
    for i, _path in enumerate(fplist):
        acc_squared = [(abs(a_s**2 + a_d**2)) for (a_s, a_d) in zip(_path.s_dd, _path.d_dd)]

        if any([v > V_MAX for v in _path.s_d]):  # Max speed check
            v_d = [v for v in _path.s_d]
            
            print("rejected due to speed ", v_d)
            continue
        elif any([acc > ACC_MAX**2 for acc in acc_squared]):
            a_d = [a for a in acc_squared]

            print("rejected due to acceleration ", a_d)

            continue
        elif any([abs(kappa) > K_MAX for kappa in fplist[i].kappa]):  # Max curvature check
            # print("fplist.kappa",fplist[i].kappa)
            print("rejected due to curvature")

            continue
        # elif collision_check(_path, obs, mapx, mapy, maps):
        #     continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps):
    fplist = calc_frenet_paths(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd)
    fplist = calc_global_paths(fplist, mapx, mapy, maps)

    fplist = check_path(fplist, obs, mapx, mapy, maps)
    # find minimum cost path
    min_cost = float("inf")
    opt_traj = None
    opt_ind = 0
    for fp in fplist:
        if min_cost >= fp.c_tot:
            min_cost = fp.c_tot
            opt_traj = fp
            _opt_ind = opt_ind
        opt_ind += 1

    try:
        _opt_ind
    except NameError:
        print(" No solution ! ")

    return fplist, _opt_ind


"""
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
def quaternion2euler(x,y,z,w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

# def test():
#     # map waypoints
#     # mapx = map_center[:,0]
#     # mapy = map_center[:,1]


#     mapx = np.array([-14.745906829833984, -19.97208023071289, -19.97208023071289, -20.971765518188477, -21.971450805664062, -22.971134185791016, -23.97081756591797, -24.970502853393555, -25.97018814086914, -26.969871520996094, -27.969554901123047, -28.969240188598633, -29.96892547607422, -30.968608856201172, -31.96829605102539, -32.96797561645508, -33.9676628112793, -34.96734619140625, -35.9670295715332, -36.96671676635742, -37.966400146484375, -38.96608352661133, -39.96577072143555, -40.965450286865234, -41.96513748168945, -42.964820861816406, -43.96450424194336, -44.96419143676758, -45.96387481689453, -46.963558197021484, -47.9632453918457, -48.962928771972656, -49.96261215209961, -51.05082702636719, -52.243431091308594, -53.43478775024414, -54.6233024597168, -55.80736541748047, -56.98537063598633, -58.1557502746582, -59.31691360473633, -60.46730422973633, -61.60536193847656, -62.72956466674805, -63.8383903503418, -64.93035125732422, -66.00396728515625, -67.05779266357422, -68.09040069580078, -69.10041809082031, -70.08647918701172, -71.04724884033203, -71.98143768310547, -72.88777923583984, -73.76505279541016, -74.6120834350586, -75.42772674560547, -76.21088409423828, -76.96050262451172, -77.6755599975586, -78.3551025390625, -78.99822235107422, -79.60403442382812, -80.17173767089844, -80.7005615234375, -81.18978881835938, -81.63876342773438, -82.04688262939453, -82.4135971069336, -82.7384033203125, -83.02088165283203, -83.26063537597656, -83.45735168457031, -83.61075592041016, -83.72064208984375, -83.78732299804688, -83.84437561035156, -83.90141296386719, -83.95846557617188, -84.0155029296875, -84.07255554199219, -84.12959289550781, -84.1866455078125, -84.24368286132812, -84.30073547363281, -84.35777282714844, -84.41488647460938, -84.4712905883789, -84.52566528320312, -84.57805633544922, -84.62837982177734, -84.6767349243164, -84.72305297851562, -84.767333984375, -84.80961608886719, -84.84986114501953, -84.88809967041016, -84.92433166503906, -84.97129821777344, -84.97129821777344, -85.00269317626953, -85.03205108642578, -85.05940246582031, -85.08475494384766, -85.10806274414062, -85.12934875488281, -85.14862060546875, -85.16585540771484, -85.18109130859375, -85.19429016113281, -85.20548248291016, -85.21464538574219, -85.22177124023438, -85.22688293457031, -85.22999572753906, -85.2310791015625, -85.2301254272461, -85.22754669189453, -85.22481536865234, -85.22209167480469, -85.2193603515625, -85.21663665771484, -85.21390533447266, -85.211181640625, -85.20845031738281, -85.20567321777344, -85.20567321777344, -85.20294952392578, -85.2002182006836, -85.19749450683594, -85.19476318359375, -85.1920394897461, -85.1893081665039, -85.18658447265625, -85.18385314941406, -85.1811294555664, -85.17839813232422, -85.17567443847656, -85.17294311523438, -85.17021942138672, -85.16748809814453, -85.16476440429688, -85.16204071044922, -85.15930938720703, -85.15658569335938, -85.15385437011719, -85.15113067626953, -85.14839935302734, -85.14567565917969, -85.1429443359375, -85.14022064208984, -85.13748931884766, -85.134765625, -85.13203430175781, -85.12931060791016, -85.12657928466797, -85.12385559082031, -85.12113189697266, -85.11840057373047, -85.11567687988281, -85.11294555664062, -85.11022186279297, -85.10749053955078, -85.10476684570312, -85.10203552246094, -85.09931182861328, -85.0965805053711, -85.09385681152344, -85.09112548828125, -85.0884017944336, -85.0856704711914, -85.08294677734375, -85.0802230834961, -85.0774917602539, -85.07476806640625, -85.07203674316406, -85.0693130493164, -85.06658172607422, -85.06385803222656, -85.06112670898438, -85.05840301513672, -85.05567169189453, -85.05294799804688, -85.05021667480469, -85.04749298095703, -85.04476165771484, -85.04203796386719, -85.03931427001953, -85.03658294677734, -85.03385925292969, -85.0311279296875, -85.02840423583984, -85.02567291259766, -85.02294921875, -85.02021789550781, -85.01749420166016, -85.01476287841797, -85.01203918457031, -85.00930786132812, -85.00658416748047, -85.00385284423828, -85.00112915039062, -84.99839782714844, -84.99567413330078, -84.99295043945312, -84.99021911621094, -84.98749542236328, -84.9847640991211, -84.98204040527344, -84.97930908203125, -84.9765853881836, -84.9738540649414, -84.97113037109375, -84.96839904785156, -84.9656753540039, -84.96294403076172, -84.96022033691406, -84.95748901367188, -84.95476531982422, -84.95204162597656, -84.94931030273438, -84.94658660888672, -84.94385528564453, -84.94113159179688, -84.93840026855469, -84.93567657470703, -84.93294525146484, -84.93022155761719, -84.927490234375, -84.92476654052734, -84.92203521728516, -84.9193115234375, -84.91658020019531, -84.91385650634766, -84.9111328125, -84.90840148925781, -84.90567779541016, -84.90294647216797, -84.90022277832031, -84.89749145507812, -84.89337921142578, -84.89337921142578, -84.89065551757812, -84.88792419433594, -84.8370361328125, -84.58683776855469, -84.13280487060547, -83.48725891113281, -82.66769409179688, -81.69635772705078, -80.59959411621094, -79.40715026855469, -78.18250274658203, -77.1763916015625, -76.16060638427734, -75.14285278320312, -74.12334442138672, -73.10230255126953, -72.07994079589844, -71.05648803710938, -70.03215789794922, -69.0071792602539, -67.98176574707031, -66.96573638916016, -65.05867767333984, -65.05867767333984, -64.0586929321289, -63.058712005615234, -62.0587272644043, -61.058738708496094, -60.05875015258789, -59.05876541137695, -58.058780670166016, -57.05879211425781, -56.05880355834961, -55.05881881713867, -54.05883026123047, -53.05884552001953, -52.05885696411133, -51.05887222290039, -50.05888366699219, -49.05889892578125, -48.05891036987305, -47.05892562866211, -46.058937072753906, -45.0589485168457, -43.36897277832031, -43.36897277832031, -42.368988037109375, -41.36899948120117, -41.36899948120117, -40.41107940673828, -39.457340240478516, -38.50496292114258, -37.554588317871094, -36.60685729980469, -35.66240310668945, -34.72186279296875, -33.78587341308594, -32.85506057739258, -31.9300479888916, -31.011463165283203, -30.09992218017578, -29.26520538330078, -28.45561408996582, -27.671310424804688, -26.91575050354004, -26.192277908325195, -25.504085540771484, -24.854204177856445, -24.24551010131836, -23.68069076538086, -23.1622371673584, -22.692440032958984, -22.241901397705078, -21.769424438476562, -21.25213050842285, -21.25213050842285, -20.717601776123047, -20.138717651367188, -20.138717651367188, -19.516799926757812, -18.548507690429688, -18.548507690429688, -17.827720642089844, -17.08795166015625, -17.08795166015625, -16.454469680786133, -15.852869987487793, -15.284481048583984, -14.750556945800781, -14.252274513244629, -13.79073715209961, -13.366958618164062, -12.981878280639648, -12.636343002319336, -12.331117630004883, -12.066873550415039, -11.844197273254395, -11.66220474243164, -11.484597206115723, -11.15658950805664, -11.15658950805664, -10.978103637695312, -10.978103637695312, -10.828479766845703, -10.692384719848633, -10.569842338562012, -10.46088981628418, -10.365540504455566, -10.283818244934082, -10.215730667114258, -10.161298751831055, -10.120522499084473, -10.093425750732422, -10.079814910888672, -10.079814910888672, -10.073488235473633, -10.067159652709961, -10.067159652709961, -10.060832977294922, -10.054505348205566, -10.048177719116211, -10.041851043701172, -10.035523414611816, -10.029195785522461, -10.022869110107422, -10.01654052734375, -10.010213851928711, -10.003886222839355, -9.99755859375, -9.991231918334961, -9.984904289245605, -9.97857666015625, -9.972249984741211, -9.965921401977539, -9.9595947265625, -9.953267097473145, -9.946939468383789, -9.94061279296875, -9.934285163879395, -9.927957534790039, -9.921630859375, -9.915302276611328, -9.908975601196289, -9.902647972106934, -9.896320343017578, -9.889993667602539, -9.883666038513184, -9.877338409423828, -9.871011734008789, -9.864684104919434, -9.858356475830078, -9.852028846740723, -9.845701217651367, -9.839374542236328, -9.833046913146973, -9.826719284057617, -9.820392608642578, -9.814064979553223, -9.807737350463867, -9.801410675048828, -9.795082092285156, -9.788755416870117, -9.782427787780762, -9.776100158691406, -9.769773483276367, -9.763445854187012, -9.757118225097656, -9.750791549682617, -9.744462966918945, -9.738136291503906, -9.73180866241455, -9.725481033325195, -9.719154357910156, -9.7128267288208, -9.706499099731445, -9.700172424316406, -9.693843841552734, -9.687517166137695, -9.68118953704834, -9.674861907958984, -9.668535232543945, -9.66220760345459, -9.655879974365234, -9.649553298950195, -9.643224716186523, -9.636898040771484, -9.630570411682129, -9.624242782592773, -9.617916107177734, -9.611588478088379, -9.605260848999023, -9.594377517700195, -9.594377517700195, -9.588050842285156, -9.5817232131958, -9.575395584106445, -9.569068908691406, -9.56274127960205, -9.556413650512695, -9.55008602142334, -9.543758392333984, -9.537431716918945, -9.53110408782959, -9.524776458740234, -9.518449783325195, -9.51212215423584, -9.505794525146484, -9.499467849731445, -9.493139266967773, -9.486812591552734, -9.480484962463379, -9.474157333374023, -9.467830657958984, -9.461503028869629, -9.455175399780273, -9.448848724365234, -9.442520141601562, -9.436193466186523, -9.429865837097168, -9.423538208007812, -9.417211532592773, -9.410883903503418, -9.39924144744873, -9.39924144744873, -9.392913818359375, -9.386587142944336, -9.380258560180664, -9.373931884765625, -9.36760425567627, -9.361276626586914, -9.351849555969238, -9.351849555969238, -9.345521926879883, -9.339194297790527, -9.332867622375488, -9.334887504577637, -9.391243934631348, -9.508682250976562, -9.686627388000488, -9.924208641052246, -10.22026252746582, -10.545626640319824, -10.948944091796875, -11.440156936645508, -12.008058547973633, -12.639690399169922, -13.356975555419922, -14.302530288696289, -15.24808406829834, -16.193639755249023, -17.139192581176758, -18.084747314453125, 
#                     -19.030302047729492, -19.975854873657227, -20.921409606933594, -22.46649169921875, -22.46649169921875, -23.384519577026367, -24.297531127929688, -25.2053165435791, -26.107669830322266, -27.00438117980957, -27.895248413085938, -28.780061721801758, -29.658618927001953, -30.530719757080078, -31.396160125732422, -32.25474548339844, -33.10627746582031, -33.950557708740234, -34.78739547729492, -35.61659240722656, -37.21973419189453])
#     mapy = np.array([208.37806701660156, 205.00830078125, 205.00830078125, 205.03343200683594, 205.0585479736328, 205.0836639404297, 205.10879516601562, 205.1339111328125, 205.15904235839844, 205.1841583251953, 205.20928955078125, 205.23440551757812, 205.259521484375, 205.28465270996094, 205.3097686767578, 205.33489990234375, 205.36001586914062, 205.3851318359375, 205.41026306152344, 205.4353790283203, 205.46051025390625, 205.48562622070312, 205.5107421875, 205.53587341308594, 205.5609893798828, 205.58612060546875, 205.61123657226562, 205.6363525390625, 205.66148376464844, 205.6865997314453, 205.71173095703125, 205.73684692382812, 205.76197814941406, 205.78468322753906, 205.77267456054688, 205.7168731689453, 205.61741638183594, 205.47438049316406, 205.28797912597656, 205.05848693847656, 204.7861785888672, 204.471435546875, 204.11468505859375, 203.71640014648438, 203.27713012695312, 202.79745483398438, 202.2780303955078, 201.71954345703125, 201.12277221679688, 200.48849487304688, 199.8175506591797, 199.1108856201172, 198.3694305419922, 197.59420776367188, 196.7862091064453, 195.9465789794922, 195.076416015625, 194.17691040039062, 193.24923706054688, 192.29470825195312, 191.31459045410156, 190.31015014648438, 189.28280639648438, 188.23391723632812, 187.16490173339844, 186.0771942138672, 184.97227478027344, 183.85159301757812, 182.71670532226562, 181.56912231445312, 180.41038513183594, 179.2420654296875, 178.06573486328125, 176.8829803466797, 175.6953887939453, 174.6090545654297, 173.6106719970703, 172.6123046875, 171.6139373779297, 170.61557006835938, 169.6171875, 168.6188201904297, 167.62045288085938, 166.62208557128906, 165.6237030029297, 164.62533569335938, 163.62576293945312, 162.61683654785156, 161.60780334472656, 160.59864807128906, 159.5894012451172, 158.58006286621094, 157.5706329345703, 156.56109619140625, 155.55149841308594, 154.5417938232422, 153.53201293945312, 152.5221710205078, 151.1183624267578, 151.11834716796875, 150.10833740234375, 149.0982666015625, 148.088134765625, 147.0779571533203, 146.06773376464844, 145.0574493408203, 144.04713439941406, 143.0367889404297, 142.02639770507812, 141.01597595214844, 140.0055389404297, 138.99508666992188, 137.984619140625, 136.97412109375, 135.963623046875, 134.95314025878906, 133.942626953125, 132.93846130371094, 131.93846130371094, 130.93846130371094, 129.9384765625, 128.9384765625, 127.93848419189453, 126.93849182128906, 125.93849182128906, 124.91848754882812, 124.91848754882812, 123.91848754882812, 122.91849517822266, 121.91849517822266, 120.91850280761719, 119.91850280761719, 118.91851043701172, 117.91851043701172, 116.91851806640625, 115.91851806640625, 114.91852569580078, 113.91852569580078, 112.91853332519531, 111.91853332519531, 110.91854095458984, 109.91854095458984, 108.91854858398438, 107.91854858398438, 106.9185562133789, 105.9185562133789, 104.91856384277344, 103.91856384277344, 102.91857147216797, 101.91857147216797, 100.9185791015625, 99.9185791015625, 98.91858673095703, 97.91858673095703, 96.91859436035156, 95.91859436035156, 94.9186019897461, 93.9186019897461, 92.9186019897461, 91.91860961914062, 90.91861724853516, 89.91861724853516, 88.91861724853516, 87.91862487792969, 86.91863250732422, 85.91863250732422, 84.91863250732422, 83.91864013671875, 82.91864776611328, 81.91864776611328, 80.91864776611328, 79.91865539550781, 78.91866302490234, 77.91866302490234, 76.91866302490234, 75.91867065429688, 74.9186782836914, 73.9186782836914, 72.9186782836914, 71.91868591308594, 70.91869354248047, 69.91869354248047, 68.91869354248047, 67.918701171875, 66.91870880126953, 65.91870880126953, 64.91870880126953, 63.91871643066406, 62.91872024536133, 61.918724060058594, 60.91872787475586, 59.91873550415039, 58.91873550415039, 57.91874313354492, 56.91874313354492, 55.91875076293945, 54.91875076293945, 53.918758392333984, 52.918758392333984, 51.918766021728516, 50.918766021728516, 49.91877365112305, 48.91877365112305, 47.91878128051758, 46.91878128051758, 45.91878890991211, 44.91878890991211, 43.91878890991211, 42.91879653930664, 41.91879653930664, 40.91880416870117, 39.91880416870117, 38.9188117980957, 37.9188117980957, 36.918819427490234, 35.918819427490234, 34.918827056884766, 33.918827056884766, 32.9188346862793, 31.918834686279297, 30.918842315673828, 29.918842315673828, 28.91884994506836, 27.91884994506836, 26.91885757446289, 25.91885757446289, 24.918865203857422, 23.918865203857422, 22.918872833251953, 21.918872833251953, 20.918880462646484, 19.918880462646484, 18.918888092041016, 17.918888092041016, 16.918895721435547, 15.91889476776123, 14.918902397155762, 13.918902397155762, 12.918910026550293, 11.918910026550293, 10.408919334411621, 10.408919334411621, 9.408923149108887, 8.408926963806152, 7.217222690582275, 5.9546918869018555, 4.750350475311279, 3.6368651390075684, 2.644439697265625, 1.7999920845031738, 1.126427412033081, 0.6420164108276367, 0.363541841506958, 0.208573579788208, 0.06662476062774658, -0.0604093074798584, -0.17250514030456543, -0.269639253616333, -0.3517957925796509, -0.41894376277923584, -0.4710829257965088, -0.5081976652145386, -0.5302726030349731, -0.5383607149124146, -0.5481665134429932, -0.5481665134429932, -0.553308367729187, -0.5584502220153809, -0.5635920763015747, -0.5687339305877686, -0.5738757848739624, -0.5790176391601562, -0.5841594934463501, -0.589301347732544, -0.5944432020187378, -0.5995850563049316, -0.6047269105911255, -0.6098687648773193, -0.6150106191635132, -0.620152473449707, -0.6252943277359009, -0.6304360628128052, -0.635577917098999, -0.6407197713851929, -0.6458616256713867, -0.6510034799575806, -0.6596932411193848, -0.6596932411193848, -0.6648350954055786, -0.6699769496917725, -0.6699769496917725, -0.6854658126831055, -0.7256138324737549, -0.7904846668243408, -0.8800265789031982, -0.9941892623901367, -1.13289213180542, -1.2960411310195923, -1.483526349067688, -1.6952245235443115, -1.930985450744629, -2.1906607151031494, -2.474071979522705, -2.7745187282562256, -3.128350257873535, -3.5351665019989014, -3.9931697845458984, -4.500338077545166, -5.054433345794678, -5.653007507324219, -6.293419361114502, -6.97283935546875, -7.688269138336182, -8.43655014038086, -9.25943374633789, -10.140777587890625, -11.105731010437012, -11.105731010437012, -12.048114776611328, -12.963922500610352, -12.963922500610352, -13.851066589355469, -15.074146270751953, -15.074146270751953, -15.883016586303711, -16.640962600708008, -16.640962600708008, -17.304941177368164, -17.997941970825195, -18.718425750732422, -19.464815139770508, -20.235454559326172, -21.028642654418945, -21.84263038635254, -22.675628662109375, -23.525787353515625, -24.3912410736084, -25.270071029663086, -26.160341262817383, -27.06683349609375, -28.050933837890625, -29.869403839111328, -29.869403839111328, -30.917556762695312, -30.917556762695312, -31.881460189819336, -32.84737014770508, -33.81509780883789, -34.78443908691406, -35.75521469116211, -36.727237701416016, -37.7003059387207, -38.674232482910156, -39.64883041381836, -40.62390899658203, -41.6285285949707, -41.6285285949707, -42.628509521484375, -43.62849044799805, -43.62849044799805, -44.62847137451172, -45.62845230102539, -46.6284294128418, -47.62841033935547, -48.62839126586914, -49.62836837768555, -50.62834930419922, -51.62833023071289, -52.62831115722656, -53.628292083740234, -54.62826919555664, -55.62825012207031, -56.628231048583984, -57.62820816040039, -58.62818908691406, -59.628170013427734, -60.628150939941406, -61.62813186645508, -62.628108978271484, -63.628089904785156, -64.62806701660156, -65.62804412841797, -66.6280288696289, -67.62800598144531, -68.62798309326172, -69.62796783447266, -70.62794494628906, -71.62792205810547, -72.6279067993164, -73.62788391113281, -74.62786865234375, -75.62784576416016, -76.6278305053711, -77.6278076171875, -78.6277847290039, -79.62776947021484, -80.62774658203125, -81.62772369384766, -82.6277084350586, -83.627685546875, -84.6276626586914, -85.62764739990234, -86.62762451171875, -87.62760162353516, -88.6275863647461, -89.62757110595703, -90.6275405883789, -91.62752532958984, -92.62751007080078, -93.62748718261719, -94.6274642944336, -95.62744903564453, -96.62742614746094, -97.62740325927734, -98.62738800048828, -99.62736511230469, -100.6273422241211, -101.62732696533203, -102.62730407714844, -103.62728118896484, -104.62726593017578, -105.62725067138672, -106.6272201538086, -107.62720489501953, -108.62718200683594, -109.62716674804688, -110.62714385986328, -111.62712860107422, -112.62710571289062, -113.62708282470703, -114.62706756591797, -115.62704467773438, -116.62702178955078, -118.34699249267578, -118.34699249267578, -119.34696960449219, -120.34695434570312, -121.34693145751953, -122.34691619873047, -123.34689331054688, -124.34687042236328, -125.34685516357422, -126.34683227539062, -127.34680938720703, -128.3468017578125, -129.34678649902344, -130.3467559814453, -131.34674072265625, -132.3467254638672, -133.34669494628906, -134.3466796875, -135.34666442871094, -136.3466339111328, -137.34661865234375, -138.3466033935547, -139.34658813476562, -140.3465576171875, -141.34654235839844, -142.34652709960938, -143.34649658203125, -144.3464813232422, -145.34646606445312, -146.346435546875, -147.34642028808594, -149.18638610839844, -149.18638610839844, -150.18637084960938, -151.18634033203125, -152.1863250732422, -153.18630981445312, -154.186279296875, -155.18626403808594, -156.6762237548828, -156.6762237548828, -157.67620849609375, -158.67617797851562, -159.67616271972656, -160.61378479003906, -161.4893341064453, -162.35879516601562, -163.21792602539062, -164.06248474121094, -164.88839721679688, -165.59544372558594, -166.20956420898438, -166.7559051513672, -167.2220458984375, -167.5973358154297, -167.88697814941406, -168.2124481201172, -168.5379180908203, -168.86337280273438, -169.1888427734375, -169.51429748535156, -169.8397674560547, -170.1652374267578, -170.49069213867188, 
#                     -171.02255249023438, -171.02255249023438, -171.34634399414062, -171.68402099609375, -172.03549194335938, -172.4006805419922, -172.779541015625, -173.17193603515625, -173.5778045654297, -173.99703979492188, -174.4295196533203, -174.87521362304688, -175.33395385742188, -175.8056640625, -176.29022216796875, -176.78753662109375, -177.2974853515625, -178.33367919921875])
   
#     # mapx = np.array(mapx)   
#     # mapy = np.array(mapy)

#     # print("mapx",mapx)
#     # mapx = np.arange(0,20,0.1)
#     # mapy = np.arange(0,20,0.1)

#     # static obstacles
#     obs = np.array([[13, WIDTH],
#                    [80, -WIDTH],
#                    [150, WIDTH],
#                    [285, -WIDTH]
#                    ])

#     # get maps
#     maps = np.zeros(mapx.shape)
#     for i in range(len(mapx)-1):
#         x = mapx[i]
#         y = mapy[i]
#         sd = get_frenet(x, y, mapx, mapy)
#         maps[i] = sd[0]
#         # print("x {}, y {}, s{}".format(x,y,maps[i]))
#     # get global position info. of static obstacles
#     obs_global = np.zeros(obs.shape)
#     for i in range(len(obs[:,0])):
#         _s = obs[i,0]
#         _d = obs[i,1]
#         # print("_s {}, _d {}".format(_s,_d))
#         xy = get_cartesian(_s, _d, mapx, mapy, maps)
#         obs_global[i] = xy[:-1]

#     # 자챠량 관련 initial condition
#     x = -15
#     y = 207
#     yaw = 0
#     v = 0
#     a = 0

#     s, d = get_frenet(x, y, mapx, mapy);
#     # print("s {}, d {}".format(s,d))
#     x, y, yaw_road = get_cartesian(s, d, mapx, mapy, maps)
#     yawi = yaw - yaw_road

#     # s 방향 초기조건
#     si = s
#     si_d = v*np.cos(yawi)
#     si_dd = a*np.cos(yawi)
#     sf_d = TARGET_SPEED
#     sf_dd = 0

#     # d 방향 초기조건
#     di = d
#     di_d = v*np.sin(yawi)
#     di_dd = a*np.sin(yawi)
#     df_d = 0
#     df_dd = 0

#     # opt_d = di

#     # 시뮬레이션 수행 (SIM_STEP 만큼)
#     plt.figure(figsize=(7,10))
#     for step in range(SIM_STEP):

#         # optimal planning 수행 (output : valid path & optimal path index)
#         path, opt_ind = frenet_optimal_planning(si, si_d, si_dd,
#                                                 sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps)

#         '''
#         다음 시뮬레이션 step 에서 사용할 initial condition update.
#         본 파트에서는 planning 만 수행하고 control 은 따로 수행하지 않으므로,
#         optimal trajectory 중 현재 위치에서 한개 뒤 index 를 다음 step 의 초기초건으로 사용.
#         '''
#         # si = path[0].s[1]
#         # si_d = path[0].s_d[1]
#         # si_dd = path[0].s_dd[1]
#         # di = path[0].d[1]
#         # di_d = path[0].d_d[1]
#         # di_dd = path[0].d_dd[1]
#         si = path[opt_ind].s[1]
#         si_d = path[opt_ind].s_d[1]
#         si_dd = path[opt_ind].s_dd[1]
#         di = path[opt_ind].d[1]
#         di_d = path[opt_ind].d_d[1]
#         di_dd = path[opt_ind].d_dd[1]
#         # consistency cost를 위해 update
#         # opt_d = path[opt_ind].d[-1]

#         if SHOW_ANIMATION:  # pragma: no cover

#             plt.cla()
#             # for stopping simulation with the esc key.
#             plt.gcf().canvas.mpl_connect(
#                 'key_release_event',
#                 lambda event: [exit(0) if event.key == 'escape' else None])

#             # plt.plot(map_center[:,0], map_center[:,1], 'k', linewidth=2)
#             plt.plot(mapx, mapy, 'k', linewidth=2)

#             # plt.plot(map_in[:,0], map_in[:,1], 'k', linewidth=2)
#             # plt.plot(map_out[:,0], map_out[:,1], 'k', linewidth=2)
#             # plt.plot(wp_in[:,0], wp_in[:,1], color='slategray', linewidth=2, alpha=0.5)
#             # plt.plot(wp_out[:,0], wp_out[:,1], color='slategray', linewidth=2, alpha=0.5)

#             # plot obstacle
#             for ob in obs_global:
#                 plt.plot(ob[0], ob[1], "s", color="crimson", MarkerSize=15, alpha=0.6)

#             for i in range(len(path)):
#                     plt.plot(path[i].x, path[i].y, "-", color="crimson", linewidth=1.5, alpha=0.6)

#             plt.plot(path[opt_ind].x, path[opt_ind].y, "o-", color="dodgerblue", linewidth=3)

#             # plot car
#             plot_car(path[opt_ind].x[0], path[opt_ind].y[0], path[opt_ind].yaw[0], steer=0)

#             plt.axis('equal')
#             plt.title("[Simulation] v : " + str(si_d)[0:4] + " m/s")
#             plt.grid(True)
#             plt.xlabel("X [m]")
#             plt.ylabel("Y [m]")

#             plt.pause(0.01)
#             # input("Press enter to continue...")


# if __name__ == "__main__":
#     test()

