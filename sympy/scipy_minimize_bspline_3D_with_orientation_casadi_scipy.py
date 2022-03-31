from math import sin, cos
from re import L
import numpy as np
from numpy import linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import BSpline
from scipy import optimize
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, summation, Piecewise
# from sympy.functions.special.bsplines import bspline_basis, _add_splines

from casadi import *

from scipy_minimize_bspline_3D_with_orientation import computer_Rb_W

class Trajectory:

    def __init__(self, np=12, nyaw=4, dp=4, dyaw=2) -> None:
        self.n_p = np # number of control points is (n+1)
        self.n_yaw = nyaw
        self.d_p = dp
        self.d_yaw = dyaw

        self.t = MX.sym('t')
        self.T = MX.sym('T')

        self.Cx = MX.sym('Cx', self.n_p+1)
        self.Cy = MX.sym('Cy', self.n_p+1)
        self.Cz = MX.sym('Cz', self.n_p+1)
        self.Cyaw = MX.sym('Cyaw', self.n_yaw+1)

        self.x = vertcat(self.Cx, self.Cy, self.Cz, self.Cyaw, self.T)

        self.cpXlen = (self.n_p + 1) 
        self.cpYlen = (self.n_p + 1) 
        self.cpZlen = (self.n_p + 1) 
        self.cpYawlen = self.n_yaw + 1

        self.xlen = (self.n_p + 1) * 3 + (self.n_yaw + 1) + 1
        

        knots_p = createNormalizedClampKnotVector(self.n_p+1, self.d_p).tolist()
        knots_yaw = createNormalizedClampKnotVector(self.n_yaw+1, self.d_yaw).tolist()

        dimention = 1
        self.Px = bspline(self.t/self.T, self.Cx, [knots_p], [self.d_p], dimention, {})
        self.Py = bspline(self.t/self.T, self.Cy, [knots_p], [self.d_p], dimention, {})
        self.Pz = bspline(self.t/self.T, self.Cz, [knots_p], [self.d_p], dimention, {})
        self.Pyaw = bspline(self.t/self.T, self.Cyaw, [knots_yaw], [self.d_yaw], dimention, {})


        ## computing position time gradients
        self.Px_dt_list = [self.Px]
        self.Py_dt_list = [self.Py]
        self.Pz_dt_list = [self.Pz]
        for _ in range(self.d_p):
            self.Px_dt_list.append(gradient(self.Px_dt_list[-1], self.t))
            self.Py_dt_list.append(gradient(self.Py_dt_list[-1], self.t))
            self.Pz_dt_list.append(gradient(self.Pz_dt_list[-1], self.t))

        ## computing heading time gradients
        self.Pyaw_dt_list = [self.Pyaw]
        for _ in range(self.d_yaw):
            self.Pyaw_dt_list.append(gradient(self.Pyaw_dt_list[-1], self.t))
 

        self.P_dt_fun = {}
        P_dt = {'X': self.Px_dt_list, 
                'Y': self.Py_dt_list,
                'Z': self.Pz_dt_list}
        cp = { 'X': self.Cx,
                'Y': self.Cy,
                'Z': self.Cz}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            Pj_dt_fun_list = []
            for j in range(self.d_p+1):
                fun = Function('P{}_d{}t_fun'.format(axis, j), [self.t, self.T, cp[axis]], [P_dt[axis][j]])
                Pj_dt_fun_list.append(fun)
            self.P_dt_fun[axis] = Pj_dt_fun_list
        Pyaw_dt_fun_list = []
        for j in range(self.d_yaw+1):
            fun = Function('Pyaw_d{}t_fun'.format(j), [self.t, self.T, self.Cyaw], [self.Pyaw_dt_list[j]])
            Pyaw_dt_fun_list.append(fun)
        self.P_dt_fun['Yaw'] = Pyaw_dt_fun_list

        self.indices = {
                'X': (0, self.cpXlen),
                'Y': (self.cpXlen, self.cpXlen+self.cpYlen),
                'Z': (self.cpXlen+self.cpYlen, self.cpXlen+self.cpYlen+self.cpZlen),
                'Yaw': (self.cpXlen+self.cpYlen+self.cpZlen, self.cpXlen+self.cpYlen+self.cpZlen+self.cpYawlen)
            }

    def getC(self, x, axis):
        i = self.indices[axis]
        return x[i[0]:i[1]]


def createNormalizedClampKnotVector(n, k):
    t=np.linspace(0,1,n-(k-1),endpoint=True)
    # t=np.linspace(0,1,n-k+2,endpoint=True)
    t=np.append([0]*k,t)
    t=np.append(t,[1]*k)
    return t


def objectiveFunction(x, numT, mC_mag_dot_fun, traj: Trajectory, m_W, ti_list, alpha):
    T = x[-numT:].sum()
    avg_mag = []
    for ti in ti_list:
        mag = mC_mag_dot_fun(ti*T, T, traj.getC(x, 'X'), traj.getC(x, 'Y'), traj.getC(x, 'Z'), traj.getC(x, 'Yaw'), m_W)
        avg_mag.append(mag)
    avg_mag = np.array(avg_mag)
    return T + alpha * avg_mag.mean() 


def start_constraint(x, traj: Trajectory, axis, i, cond):
    '''
        equality constraint:
        diP_dt(0) = cond
        diP_dt(0) - cond = 0
    '''
    indices = {
            'X': (0, traj.cpXlen),
            'Y': (traj.cpXlen, traj.cpXlen+traj.cpYlen),
            'Z': (traj.cpXlen+traj.cpYlen, traj.cpXlen+traj.cpYlen+traj.cpZlen),
            'Yaw': (traj.cpXlen+traj.cpYlen+traj.cpZlen, traj.cpXlen+traj.cpYlen+traj.cpZlen+traj.cpYawlen)
    }
    return cond - traj.P_dt_fun[axis][i](0.001, x[-1], x[indices[axis][0]:indices[axis][1]])

def end_constraint(x, traj: Trajectory, axis, i, cond):
    '''
        equality constraint:
        diP_dt(t[j]) = cond
        diP_dt(t[j]) - cond = 0
    '''
    indices = {
            'X': (0, traj.cpXlen),
            'Y': (traj.cpXlen, traj.cpXlen+traj.cpYlen),
            'Z': (traj.cpXlen+traj.cpYlen, traj.cpXlen+traj.cpYlen+traj.cpZlen),
            'Yaw': (traj.cpXlen+traj.cpYlen+traj.cpZlen, traj.cpXlen+traj.cpYlen+traj.cpZlen+traj.cpYawlen)
    }
    T = x[-1]
    return cond - traj.P_dt_fun[axis][i](T, x[-1], x[indices[axis][0]:indices[axis][1]])

def ineq_constraint(x, traj: Trajectory, axis, i, cond, ratio, sign):
    '''
        sign * diP_dt(ratio*T) <= cond
        cond - sign * diP_dt(ratio*T) >= 0
        g >= 0
    '''
    indices = {
            'X': (0, traj.cpXlen),
            'Y': (traj.cpXlen, traj.cpXlen+traj.cpYlen),
            'Z': (traj.cpXlen+traj.cpYlen, traj.cpXlen+traj.cpYlen+traj.cpZlen),
            'Yaw': (traj.cpXlen+traj.cpYlen+traj.cpZlen, traj.cpXlen+traj.cpYlen+traj.cpZlen+traj.cpYawlen)
    }
    assert abs(sign) == 1
    T = x[-1]
    t = ratio * T
    g = cond - sign * traj.P_dt_fun[axis][i](t, x[-1], x[indices[axis][0]:indices[axis][1]])
    return g

def compute_mC(traj: Trajectory, Rc_B, Oc_B, m_W: MX):
    '''
        Rc_B is the rotation matrix from the camera frame C to the body frame B.
        Oc_B is the origin of the camera frame C expressed in the body frame B.
        m_W is the position of the visual feature in the world frame W.
        coordinate_index is an integer that must be 0 or 1, which indicates the coordinate (x or y) that the FOV constraint is going to be on.
        FOV is the angle (expressed in radian) of the field of view of the camera in the x or y direction (depends on the coordinate_index, if it's 0, the FOV is for the x axis)
    '''
    xi = traj.P_dt_fun['X'][0](traj.t, traj.T, traj.Cx)
    yi = traj.P_dt_fun['Y'][0](traj.t, traj.T, traj.Cy)
    zi = traj.P_dt_fun['Z'][0](traj.t, traj.T, traj.Cz)
    theta = traj.P_dt_fun['Yaw'][0](traj.t, traj.T, traj.Cyaw)

    xi_acc = traj.P_dt_fun['X'][2](traj.t, traj.T, traj.Cx)
    yi_acc = traj.P_dt_fun['Y'][2](traj.t, traj.T, traj.Cy)
    zi_acc = traj.P_dt_fun['Z'][2](traj.t, traj.T, traj.Cz)
    g = 9.806

    ZB = vcat([xi_acc, yi_acc, zi_acc + g]) #.reshape((-1, 1))
    ZB = ZB / norm_2(ZB)

    XC = vcat([cos(theta), sin(theta), 0]) #.reshape((-1, 1))

    YB = cross(ZB, XC, 1)
    YB = YB / norm_2(YB)

    XB = cross(YB, ZB, 1)

    ## Rb_W is the rotation matrix from the body frame B to the world frame W.
    Rb_W = hcat([XB, YB, ZB])

    ## m_C is the visual feature m represented in the camera frame C.
    ## m_C = [Rc_B]^T * [Rb_W]^T * [m_w - (Rb_W * Oc_B) - Ob_w]
    Ob_W = vcat([xi, yi, zi])

    d = m_W - mtimes(Rb_W, Oc_B) - Ob_W 

    m_C = mtimes(transpose(Rc_B), mtimes(transpose(Rb_W), d))

    ## divide m_C over its Z component.
    m_C = m_C / m_C[2]
    return m_C[:2]

def inFOV_ineq_constraint(x, traj: Trajectory, mC_fun, m_W, coordinate_index, FOV, ratio, sign):
    '''
        Rc_B is the rotation matrix from the camera frame C to the body frame B.
        Oc_B is the origin of the camera frame C expressed in the body frame B.
        m_W is the position of the visual feature in the world frame W.
        coordinate_index is an integer that must be 0 or 1, which indicates the coordinate (x or y) that the FOV constraint is going to be on.
        FOV is the angle (expressed in radian) of the field of view of the camera in the x or y direction (depends on the coordinate_index, if it's 0, the FOV is for the x axis)
    '''
    assert abs(sign) == 1
    T = x[-1]
    t = ratio * T
    m_C = mC_fun(t, T, traj.getC(x, 'X'), traj.getC(x, 'Y'), traj.getC(x, 'Z'), traj.getC(x, 'Yaw'), m_W) 
    
    if sign == 1:
        g = sin(FOV/2) - m_C[coordinate_index]
    else:
        g = sin(FOV/2) + m_C[coordinate_index]
    return g

def solve_n_D_OptimizationProblem(startPose=[-5, -30, 1, np.pi/3.], endPose=[0, -4, 2.5, np.pi/2.], featuresWorldPosition=np.array([[9.014978e-01, 1.497865e-03, 2.961498],[-8.985040e-01, 1.497627e-03, 2.961498]]), camera_FOV_h=80, camera_FOV_v=60):
    # n >= k-1, the number of control points is n+1
    # n + 1 >= k
    n_p = 16 # number of control points is (n+1)
    n_yaw = 6

    d_p = 4
    d_yaw = 2

    knots_p = createNormalizedClampKnotVector(n_p+1, d_p)
    knots_yaw = createNormalizedClampKnotVector(n_yaw+1, d_yaw)

    numT = 1
    traj = Trajectory(n_p, n_yaw, d_p, d_yaw)

    ## visual features stuff:
    FOV_h = camera_FOV_h*np.pi/180.0
    FOV_v = camera_FOV_v*np.pi/180.0
    M = featuresWorldPosition
    Rc_B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    Oc_B = np.zeros((3, 1))


    m_W = MX.sym('m_W', 3)
    mC = compute_mC(traj, Rc_B, Oc_B, m_W)
    mC_fun = Function('mC', [traj.t, traj.T, traj.Cx, traj.Cy, traj.Cz, traj.Cyaw, m_W], [mC])
    
    mC_mag = norm_2(mC) # try norm_1 or norm_infinity
    mC_mag_dot = gradient(mC_mag, traj.t)

    mC_mag_dot_fun = Function('mC_mag_dot', [traj.t, traj.T, traj.Cx, traj.Cy, traj.Cz, traj.Cyaw, m_W], [mC_mag_dot])




    x0_cpx = np.linspace(startPose[0], endPose[0], num=traj.cpXlen, endpoint=True)
    x0_cpy = np.linspace(startPose[1], endPose[1], num=traj.cpYlen, endpoint=True)
    x0_cpz = np.linspace(startPose[2], endPose[2], num=traj.cpZlen, endpoint=True)
    x0_cpyaw = np.linspace(startPose[3], endPose[3], num=traj.cpYawlen, endpoint=True)

    x0 = [0.0] * (traj.xlen)
    x0[0:traj.cpXlen] = x0_cpx
    x0[traj.cpXlen:traj.cpXlen + traj.cpYlen] = x0_cpy
    x0[traj.cpXlen + traj.cpYlen:traj.cpXlen + traj.cpYlen + traj.cpZlen] = x0_cpz
    x0[traj.cpXlen + traj.cpYlen + traj.cpZlen:traj.cpXlen + traj.cpYlen + traj.cpZlen + traj.cpYawlen] = x0_cpyaw
    T0_list = [50, 50, 50]
    for i in range(1, numT+1):
        x0[-i] = T0_list[-i] 

    # print('x0 =', x0)

    
    ### adding equality constraints:
    ## inital conditions for the position and the reset of the derivatives:
    initialConditionList_Px = [0.0] * (d_p+1)
    initialConditionList_Px[0] = startPose[0] # inital position x(0) = 1
    initialConditionList_Py = initialConditionList_Px.copy()
    initialConditionList_Py[0] = startPose[1] # inital position x(0) = 1
    initialConditionList_Pz = initialConditionList_Px.copy()
    initialConditionList_Pz[0] = startPose[2] # inital position x(0) = 1

    initalConditionList_yaw = [0.0] * (d_yaw+1)
    initalConditionList_yaw[0] = startPose[-1] # inital orientation yaw(0) = 45 degree

    constraints = []

    ## initial position constraints
    for i in range(d_p+1): # we have (d+1) conditions; the position and (d) derivatives.
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (traj, 'X', i, initialConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (traj, 'Y', i, initialConditionList_Py[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (traj, 'Z', i, initialConditionList_Pz[i])} 
        constraints.append(eq_cons)

    ## inital orientation constraints
    for i in range(d_yaw+1):
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (traj, 'Yaw', i, initalConditionList_yaw[i])} 
        constraints.append(eq_cons)

    
    ## end conditions for the position and the reset of the derivatives:
    ## there is no end conditions for the orientation
    endConditionList_Px = initialConditionList_Px.copy()
    endConditionList_Py = initialConditionList_Py.copy()
    endConditionList_Pz = initialConditionList_Pz.copy()
    endConditionList_Px[0] = endPose[0]
    endConditionList_Py[0] = endPose[1]
    endConditionList_Pz[0] = endPose[2]

    for i in range(d_p+1): 
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (traj, 'X', i, endConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (traj, 'Y', i, endConditionList_Py[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (traj, 'Z', i, endConditionList_Pz[i])} 
        constraints.append(eq_cons)

    # ## midpoints' conditions:
    # p_mid1 = (3, 5)
    # midpoint1_constx = {'type': 'eq', 'fun': minpoint_constraint, 'args': (Px, 0, p_mid1[0], numT, -2)} 
    # midpoint1_consty = {'type': 'eq', 'fun': minpoint_constraint, 'args': (Py, 0, p_mid1[1], numT, -2)} 
    # constraints.append(midpoint1_constx)
    # constraints.append(midpoint1_consty)
    # p_mid2 = (12, 10)
    # midpoint2_constx = {'type': 'eq', 'fun': minpoint_constraint, 'args': (Px, 0, p_mid2[0], numT, -1)} 
    # midpoint2_consty = {'type': 'eq', 'fun': minpoint_constraint, 'args': (Py, 0, p_mid2[1], numT, -1)} 
    # constraints.append(midpoint2_constx)
    # constraints.append(midpoint2_consty)
                                    
                            
    ### adding inequality constraints:
    positionTimeDerivativesLimitList = [2 for i in range(d_p+1)] 
    positionTimeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    positionTimeDerivativesLimitList[1] = 10 # vel
    positionTimeDerivativesLimitList[2] = 5 # acc

    orientationTimeDerivativesLimitList = [2 for i in range(d_p+1)] 
    orientationTimeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    orientationTimeDerivativesLimitList[1] = 10 # heading angualr vel limit rad/s 
    orientationTimeDerivativesLimitList[2] = 2 # heading angualr acc limit rad/s**2

    num = 20
    ti_list = np.linspace(0, 1, num, endpoint=True)
    ti_list[0] = 0.001
    for i in range(1, d_p+1): # start from 1 so you skip the position
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'X', i, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'X', i, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'Y', i, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'Y', i, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'Z', i, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'Z', i, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
    
    for i in range(1, d_yaw+1): # start from 1 so you skip the heading
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'Yaw', i, orientationTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (traj, 'Yaw', i, orientationTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
    
    # ## visual features constraints:
    for i in range(M.shape[0]):
        for ti in ti_list:
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (traj, mC_fun, M[i], 0, FOV_h, ti, 1)}
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (traj, mC_fun, M[i], 0, FOV_h, ti, -1)}
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (traj, mC_fun, M[i], 1, FOV_v, ti, 1)}
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (traj, mC_fun, M[i], 1, FOV_v, ti, -1)}
            constraints.append(ineq_cons)

    # Px = BSpline(knots_p, x0_cpx, d_p)
    # x0 = [0] * traj.xlen
    # x0[:traj.cpXlen] = x0_cpx
    # x0[-1] = 1.0
    # Px_casadi = Function('Px', [traj.t, traj.T, traj.Cx], [traj.Px])
    # # print(x0)
    # adg_list = []
    # for i in [4]:
    #     l = []
    #     for ti in np.linspace(0, 1, num=100, endpoint=True):
    #         if ti == 0:
    #             ti = 0.000001

    #         print( Px(ti, nu=i), traj.P_dt_fun['X'][i](ti, 1, x0_cpx) )
    #         d = abs( Px(ti, nu=i) - traj.P_dt_fun['X'][i](ti, 1, x0_cpx) )
    #         l.append(d)
    #     adg_list.append(l)
    # adg_list = np.array(adg_list)
    # print('--------------------------')
    # print(adg_list.mean())
    # print(adg_list.mean(axis=1))


    # print(mc_mag_fun(0.5, 1, x0_cpx, x0_cpy, x0_cpz, x0_cpyaw))
    # mC_mag_dot = gradient(mC_mag, traj.t)
    # mC_mag_dot_fun = Function('mC_mag_dot', [traj.t, traj.T, traj.Cx, traj.Cy, traj.Cz, traj.Cyaw], [mC_mag_dot])
    # print(mC_mag_dot_fun(0.5, 1, x0_cpx, x0_cpy, x0_cpz, x0_cpyaw))
    
    bounds = [(None, None)] * (traj.xlen)
    bounds[-numT:] = [(0.1, None)] * numT

    skip_opt = False
    if not skip_opt:
        print('optimization started')
        ti_list = np.linspace(0, 1, num=100, endpoint=True)
        ti_list[0] = 0.001
        alpha = 100 
        res = optimize.minimize(fun=objectiveFunction, args=(numT, mC_mag_dot_fun, traj, M[0], ti_list, alpha), x0=x0, method='SLSQP', bounds=bounds, constraints=constraints,
                        tol=1e-6, options={'maxiter': 1e8, 'disp': True})
        print(res)
        x_star = res.x
    else:
        print('optimization skipped')
        x_star = np. array([-5.00000000e+00, -5.00000000e+00, -5.00000000e+00, -5.00000000e+00,
       -5.00000000e+00, -4.35910466e+00, -3.24741307e+00, -2.56355159e+00,
       -2.50000241e+00, -2.43644678e+00, -1.75258194e+00, -6.40905659e-01,
       -1.49591519e-18,  2.32934060e-21, -1.86673985e-21, -2.70938379e-23,
       -1.74431283e-23, -3.00000000e+01, -3.00000000e+01, -3.00000000e+01,
       -3.00000000e+01, -3.00000000e+01, -2.91788168e+01, -2.66470419e+01,
       -2.23364503e+01, -1.70000000e+01, -1.16635496e+01, -7.35295803e+00,
       -4.82118318e+00, -3.99999996e+00, -3.99999997e+00, -3.99999998e+00,
       -3.99999999e+00, -4.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.32695680e+00,
        1.59794830e+00,  1.65625000e+00,  1.75000000e+00,  1.84375000e+00,
        1.90071747e+00,  2.17838014e+00,  2.49999964e+00,  2.49999966e+00,
        2.49999968e+00,  2.49999969e+00,  2.49999969e+00,  1.04719755e+00,
        1.04719755e+00,  1.04719755e+00,  1.30899694e+00,  1.39626340e+00,
        1.48352986e+00,  1.57079633e+00,  1.04062887e+01])

    T_star_list = x_star[-numT:]
    T_star = T_star_list.sum()
    print('T_star =', T_star)

    return


    t_list = np.linspace(0, T_star, num=1000, endpoint=True)
    Px_values = Px.diP_dt(x_star, 0, t_list)
    Py_values = Py.diP_dt(x_star, 0, t_list)
    Pz_values = Pz.diP_dt(x_star, 0, t_list)

    for i in range(Px.d+1):
        dPx_dt_max = Px.diP_dt(x_star, i, t_list).max()
        dPx_dt_min = Px.diP_dt(x_star, i, t_list).min()
        print('{}th Px: max={}, min={}'.format(i, dPx_dt_max, dPx_dt_min))
    for i in range(Py.d+1):
        dPy_dt_max = Py.diP_dt(x_star, i, t_list).max()
        dPy_dt_min = Py.diP_dt(x_star, i, t_list).min() 
        print('{}th Py: max={}, min={}'.format(i, dPy_dt_max, dPy_dt_min))
    for i in range(Pz.d+1):
        dPz_dt_max = Pz.diP_dt(x_star, i, t_list).max()
        dPz_dt_min = Pz.diP_dt(x_star, i, t_list).min() 
        print('{}th Pz: max={}, min={}'.format(i, dPz_dt_max, dPz_dt_min))

    # mag = 20
    # plt.plot(Px_values, Py_values, 'b:', label='minimue-time trajectory')
    # for ti in np.linspace(0, T_star, num=15, endpoint=True):
    #     x = Px.diP_dt(x_star, 0, ti)
    #     y = Py.diP_dt(x_star, 0, ti)
    #     theta = Pyaw.diP_dt(x_star, 0, ti)
    #     dx1 = mag * cos(theta+FOV/2)
    #     dy1 = mag * sin(theta+FOV/2)
    #     dx2 = mag * cos(theta-FOV/2)
    #     dy2 = mag * sin(theta-FOV/2)
    #     color = 'g' 
    #     for m in M:
    #         if isInFOV((x, y, theta), FOV, m) == False:
    #             color = 'r'
    #     plt.arrow(x, y, dx1, dy1, color=color)
    #     plt.arrow(x, y, dx2, dy2, color=color)
    # plt.plot(M[:, 0], M[:, 1], 'r*', label='tracked visual features')
    # plt.plot(initialConditionList_Px[0], initialConditionList_Py[0], 'ro', label='initial position')
    # plt.plot(endConditionList_Px[0], endConditionList_Py[0], 'go', label='end position')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.show()
    
    ## ploting in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ti_list = np.linspace(0, T_star, num=20, endpoint=True)
    x_list = Px.diP_dt(x_star, 0, ti_list) 
    y_list = Py.diP_dt(x_star, 0, ti_list) 
    z_list = Pz.diP_dt(x_star, 0, ti_list) 

    ax.plot(x_list, y_list, z_list)
    ax.scatter(x_list[0], y_list[0], z_list[0], 'o')
    ax.scatter(x_list[-1], y_list[-1], z_list[-1], 'o')
    ax.plot(M[:, 0], M[:, 1], M[:, 2], 'r*')

    Vb_list = []
    for i in range(2):
        v1 = np.array([cos(FOV_h/2), ((-1)**i)*sin(FOV_h/2), 0.0])
        Vb_list.append(v1)
        v2 = np.array([cos(FOV_v/2), 0.0, ((-1)**i)*sin(FOV_v/2)])
        Vb_list.append(v2)
    Vb_list = np.array(Vb_list).T
    print(Vb_list)

    for ti in np.linspace(0, 1, num=15, endpoint=True):
        xi = Px.diP_dt(x_star, 0, ti*T_star) 
        yi = Py.diP_dt(x_star, 0, ti*T_star) 
        zi = Pz.diP_dt(x_star, 0, ti*T_star) 
        Rb_W = computer_Rb_W(x_star, Px, Py, Pz, Pyaw, ti*T_star)
        ZBi = Rb_W[:, 0]
        Vwi = np.matmul(Rb_W, Vb_list)

        color = 'g' 
        for m in M:
            ineq1 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, m, ti, 1)
            ineq2 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, m, ti, -1)
            ineq3 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, m, ti, 1)
            ineq4 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, m, ti, -1)
            print('ti:', ti, 'ineqs:', ineq1, ineq2, ineq3, ineq4)
            if ineq1 < 0 or ineq2 < 0 or ineq3 < 0 or ineq4 < 0:
                color = 'r'
        ax.quiver(xi, yi, zi, ZBi[0], ZBi[1], ZBi[2], length=5, normalize=True, color='b')
        for k in range(4):
            ax.quiver(xi, yi, zi, Vwi[0, k], Vwi[1, k], Vwi[2, k], length=5, normalize=True, color=color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_zlim(0, 8)

    ## ploting the xy 2D plane:
    plt.figure()
    mag = 20
    plt.plot(Px_values, Py_values, 'b:', label='minimue-time trajectory')
    for ti in np.linspace(0, 1, num=15, endpoint=True):
        x = Px.diP_dt(x_star, 0, ti*T_star)
        y = Py.diP_dt(x_star, 0, ti*T_star)
        z = Pz.diP_dt(x_star, 0, ti*T_star) 
        theta = Pyaw.diP_dt(x_star, 0, ti*T_star)
        dx1 = mag * cos(theta+FOV_h/2)
        dy1 = mag * sin(theta+FOV_h/2)
        dx2 = mag * cos(theta-FOV_h/2)
        dy2 = mag * sin(theta-FOV_h/2)

        Rb_W = computer_Rb_W(x_star, Px, Py, Pz, Pyaw, ti*T_star)
        ZBi = Rb_W[:, 0]
        ZBi = mag * ZBi
        Vwi = np.matmul(Rb_W, Vb_list)
        color = 'g' 
        for m in M:
            ineq1 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, m, ti, 1)
            ineq2 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, m, ti, -1)
            ineq3 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, m, ti, 1)
            ineq4 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, m, ti, -1)
            if ineq1 < 0 or ineq2 < 0 or ineq3 < 0 or ineq4 < 0:
                color = 'r'
        plt.arrow(x, y, dx1, dy1, color=color)
        plt.arrow(x, y, dx2, dy2, color=color)
        plt.arrow(x, y, Vwi[0, 0], Vwi[1, 0], color='y')
        plt.arrow(x, y, Vwi[0, 2], Vwi[1, 2], color='y')
        plt.arrow(x, y, ZBi[0], ZBi[1], color='b')
    plt.plot(M[:, 0], M[:, 1], 'r*', label='tracked visual features')
    plt.plot(initialConditionList_Px[0], initialConditionList_Py[0], 'ro', label='initial position')
    plt.plot(endConditionList_Px[0], endConditionList_Py[0], 'go', label='end position')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    ## ploting the xz 2D plane:
    plt.figure()
    mag = 20
    plt.plot(Px_values, Pz_values, 'b:', label='minimue-time trajectory')
    for ti in np.linspace(0, 1, num=8, endpoint=True):
        x = Px.diP_dt(x_star, 0, ti*T_star)
        z = Pz.diP_dt(x_star, 0, ti*T_star) 
        theta = Pyaw.diP_dt(x_star, 0, ti*T_star)
        Rb_W = computer_Rb_W(x_star, Px, Py, Pz, Pyaw, ti*T_star)
        ZBi = Rb_W[:, 0]
        ZBi = mag * ZBi
        Vwi = np.matmul(Rb_W, Vb_list)
        color = 'g' 
        for m in M:
            ineq1 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, m, ti, 1)
            ineq2 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, m, ti, -1)
            ineq3 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, m, ti, 1)
            ineq4 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, m, ti, -1)
            if ineq1 < 0 or ineq2 < 0 or ineq3 < 0 or ineq4 < 0:
                color = 'r'
        plt.arrow(x, z, ZBi[0], ZBi[2], color='b')
        plt.arrow(x, z, Vwi[0, 1], Vwi[2, 1], color=color)
        plt.arrow(x, z, Vwi[0, 3], Vwi[2, 3], color='y')
    plt.plot(M[:, 0], M[:, 2], 'r*', label='tracked visual features')
    plt.plot(initialConditionList_Px[0], initialConditionList_Pz[0], 'ro', label='initial position')
    plt.plot(endConditionList_Px[0], endConditionList_Pz[0], 'go', label='end position')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()

    plt.show()

    # print()
    # for ti in np.linspace(0, 1, num=30, endpoint=True):
    #     ineq1 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV, M[0], ti, 1)
    #     ineq2 = inFOV_ineq_constraint(x_star, Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV, M[0], ti, -1)
    #     print(ineq1, ' ', ineq2)




    return x_star, Px, Py, Pz, Pyaw





if __name__ == '__main__':
    solve_n_D_OptimizationProblem()
