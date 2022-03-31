from math import sin, cos
import numpy as np
from numpy import linalg as la
from scipy import interpolate
from scipy import optimize
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sympy
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, summation, Piecewise, sqrt
from sympy.functions.special.bsplines import bspline_basis, _add_splines

# from sympy.vector import CoordSys3D
from sympy.physics.vector import *
from sympy.matrices import Matrix



class Traj1D:
    t = Symbol('t')
    T = Symbol('T')
    index = 0
    def __init__(self, d, n, knots, numT) -> None:
        '''
            t, T: are symbolic vars
            d: the order of B-spline
            n: the number of control points - 1
            knots: is the knot vector
        '''
        self.numT = numT
        self.d = d
        self.cpLen = n + 1
        self.C = [Symbol('C{}_{}'.format(Traj1D.index, i)) for i in range(self.cpLen)]
        Traj1D.index += 1
        self.P = createBSpline(d, knots, self.cpLen, Traj1D.t, Traj1D.T, self.C)
        self.dP_dt_list = createTimeDerivativeList(self.P, Traj1D.t, Traj1D.T, self.C)
        self.dP_dt_lambdified_List = lambdifyList(self.dP_dt_list, Traj1D.t, Traj1D.T, self.C)
    
    def setControlPointsIndices(self, start, end):
        self.start = start
        self.end = end

    def applyOptVect(self, x): 
        C = x[self.start:self.end]
        T = x[-self.numT:].sum()
        return C, T

    def diP_dt(self, x, i, t, T=None):
        '''
            call the ith dP_dt on the optVect x
        '''
        if T is None:
            C, T = self.applyOptVect(x)
        else:
            C = x[self.start:self.end]
        return self.dP_dt_lambdified_List[i](t, T, C)
    

def createNormalizedClampKnotVector(n, k):
    t=np.linspace(0,1,n-(k-1),endpoint=True)
    # t=np.linspace(0,1,n-k+2,endpoint=True)
    t=np.append([0]*k,t)
    t=np.append(t,[1]*k)
    return t

def createBSpline(d, knots, cp_len, t, T, C):
    knots = tuple(knots)
    # C = [Symbol('C{}'.format(i)) for i in range(cp_len)]
    Bd0 = bspline_basis(d, knots, 0, t/T)
    P = _add_splines(0, 0, C[0], Bd0)
    for i in range(1, cp_len):
        Bdi = bspline_basis(d, knots, i, t/T)
        P = _add_splines(1, P, C[i], Bdi)
    return P


def createTimeDerivativeList(P, t, T, C, d=4):
    diP_dt_list = [P]
    for i in range(d):
        diP_dt_list.append( diff(diP_dt_list[-1], t) )
    return diP_dt_list

def createGradList(P_list, T, C):
    P_dC_list = []
    P_dT_list = []
    for p in P_list:
        P_dC_list.append([diff(p, ci) for ci in C ])
        P_dT_list.append(diff(p, T))
    return P_dC_list, P_dT_list

def lambdifyList(l, t, T, C):
    # we need a lambdified function
    lambdified_list = []
    for f in l:
        lambdified_list.append(lambdify((t, T, C), f, 'numpy'))
    return lambdified_list

def start_constraint(x, P: Traj1D, i, cond):
    '''
        equality constraint:
        diP_dt(0) = cond
        diP_dt(0) - cond = 0
    '''
    return cond - P.diP_dt(x, i, t=0.0)

def minpoint_constraint(x, P: Traj1D, i, cond, numT, Tj):
    '''
        equality constraint:
        diP_dt(T0 + ... + Tj) = cond
        diP_dt(T0 + ... + Tj) - cond = 0
    ''' 
    assert Tj < 0
    t = x[-numT:Tj].sum()
    return cond - P.diP_dt(x, i, t)

def create_initial_sol(numT, Px, Py, Pz, Pyaw, startPose, endPose):
    x_len = Px.cpLen + Py.cpLen + Pz.cpLen + Pyaw.cpLen + numT
    x0_cpx = np.linspace(startPose[0], endPose[0], num=Px.cpLen, endpoint=True)
    x0_cpy = np.linspace(startPose[1], endPose[1], num=Py.cpLen, endpoint=True)
    x0_cpz = np.linspace(startPose[2], endPose[2], num=Pz.cpLen, endpoint=True)
    x0_cpyaw = np.linspace(startPose[3], endPose[3], num=Pyaw.cpLen, endpoint=True)

    x0 = np.zeros((x_len,))
    x0[0:Px.cpLen] = x0_cpx
    x0[Px.cpLen:Px.cpLen+Py.cpLen] = x0_cpy
    x0[Px.cpLen+Py.cpLen:Px.cpLen+Py.cpLen+Pz.cpLen] = x0_cpz
    x0[Px.cpLen + Py.cpLen+Pz.cpLen:Px.cpLen+Py.cpLen+Pz.cpLen+Pyaw.cpLen] = x0_cpyaw
    T0_list = [50, 50, 50]
    for i in range(1, numT+1):
        x0[-i] = T0_list[-i] 
    return x0


def end_constraint(x, P: Traj1D, i, cond):
    '''
        equality constraint:
        diP_dt(t[j]) = cond
        diP_dt(t[j]) - cond = 0
    '''
    _, T = P.applyOptVect(x) 
    return cond - P.diP_dt(x, i, T)

def ineq_constraint(x, P: Traj1D, i, cond, ratio, sign):
    '''
        sign * diP_dt(ratio*T) <= cond
        cond - sign * diP_dt(ratio*T) >= 0
        g >= 0
    '''
    assert abs(sign) == 1
    _, T = P.applyOptVect(x) 
    t = ratio * T
    g = cond - sign * P.diP_dt(x, i, t)
    return g

def inFOV_ineq_constraint(x, Px:Traj1D, Py:Traj1D, Pz:Traj1D, Pyaw:Traj1D, Rc_B, Oc_B, coordinate_index, FOV, m_W, ratio, sign):
    '''
        Rc_B is the rotation matrix from the camera frame C to the body frame B.
        Oc_B is the origin of the camera frame C expressed in the body frame B.
        m_W is the position of the visual feature in the world frame W.
        coordinate_index is an integer that must be 0 or 1, which indicates the coordinate (x or y) that the FOV constraint is going to be on.
        FOV is the angle (expressed in radian) of the field of view of the camera in the x or y direction (depends on the coordinate_index, if it's 0, the FOV is for the x axis)
    '''
    assert abs(sign) == 1
    _, T = Px.applyOptVect(x) 
    t = ratio * T
    xi = Px.diP_dt(x, 0, t)
    yi = Py.diP_dt(x, 0, t)
    zi = Pz.diP_dt(x, 0, t)
    theta = Pyaw.diP_dt(x, 0, t) 

    xi_acc = Px.diP_dt(x, 2, t)
    yi_acc = Py.diP_dt(x, 2, t)
    zi_acc = Pz.diP_dt(x, 2, t)
    g = 9.806

    ZB = np.array([xi_acc, yi_acc, zi_acc+g]) #.reshape((-1, 1))
    ZB = ZB / la.norm(ZB)

    XC = np.array([cos(theta), sin(theta), 0]) #.reshape((-1, 1))

    YB = np.cross(ZB, XC)
    YB = YB / la.norm(YB)

    XB = np.cross(YB, ZB)

    ## Rb_W is the rotation matrix from the body frame B to the world frame W.
    Rb_W = np.zeros((3, 3))
    Rb_W[:, 0] = XB
    Rb_W[:, 1] = YB
    Rb_W[:, 2] = ZB

    ## m_C is the visual feature m represented in the camera frame C.
    ## m_C = [Rc_B]^T * [Rb_W]^T * [m_w - (Rb_W * Oc_B) - Ob_w]
    Ob_W = np.array([xi, yi, zi]).reshape((-1, 1))

    d = m_W.reshape((-1, 1)) - np.matmul(Rb_W, Oc_B) - Ob_W 

    m_C = np.matmul(Rc_B.T, np.matmul(Rb_W.T, d))

    ## divide m_C over its Z component.
    m_C = m_C / m_C[2]
    
    if sign == 1:
        g = sin(FOV/2) - m_C[coordinate_index, 0]
    else:
        g = sin(FOV/2) + m_C[coordinate_index, 0]
    return g
    
def computer_Rb_W(x, Px:Traj1D, Py:Traj1D, Pz:Traj1D, Pyaw:Traj1D, t):
    '''
        t: the time at which Rb_W is computed. t is in [0, T]
    '''
    xi = Px.diP_dt(x, 0, t)
    yi = Py.diP_dt(x, 0, t)
    zi = Pz.diP_dt(x, 0, t)
    theta = Pyaw.diP_dt(x, 0, t) 

    xi_acc = Px.diP_dt(x, 2, t)
    yi_acc = Py.diP_dt(x, 2, t)
    zi_acc = Pz.diP_dt(x, 2, t)
    g = 9.806

    ZB = np.array([xi_acc, yi_acc, zi_acc+g]) #.reshape((-1, 1))
    ZB = ZB / la.norm(ZB)

    XC = np.array([cos(theta), sin(theta), 0]) #.reshape((-1, 1))

    YB = np.cross(ZB, XC)
    YB = YB / la.norm(YB)

    XB = np.cross(YB, ZB)

    ## Rb_W is the rotation matrix from the body frame B to the world frame W.
    Rb_W = np.zeros((3, 3))
    Rb_W[:, 0] = XB
    Rb_W[:, 1] = YB
    Rb_W[:, 2] = ZB

    return Rb_W

def computer_FOV(x, Px:Traj1D, Py:Traj1D, Pz:Traj1D, Pyaw:Traj1D, Rc_B, Oc_B, t, m_W):
    xi = Px.diP_dt(x, 0, t)
    yi = Py.diP_dt(x, 0, t)
    zi = Pz.diP_dt(x, 0, t)
    theta = Pyaw.diP_dt(x, 0, t) 

    xi_acc = Px.diP_dt(x, 2, t)
    yi_acc = Py.diP_dt(x, 2, t)
    zi_acc = Pz.diP_dt(x, 2, t)
    g = 9.806

    ZB = np.array([xi_acc, yi_acc, zi_acc+g]) #.reshape((-1, 1))
    ZB = ZB / la.norm(ZB)

    XC = np.array([cos(theta), sin(theta), 0]) #.reshape((-1, 1))

    YB = np.cross(ZB, XC)
    YB = YB / la.norm(YB)

    XB = np.cross(YB, ZB)

    ## Rb_W is the rotation matrix from the body frame B to the world frame W.
    Rb_W = np.zeros((3, 3))
    Rb_W[:, 0] = XB
    Rb_W[:, 1] = YB
    Rb_W[:, 2] = ZB

    ## m_C is the visual feature m represented in the camera frame C.
    ## m_C = [Rc_B]^T * [Rb_W]^T * [m_w - (Rb_W * Oc_B) - Ob_w]
    Ob_W = np.array([xi, yi, zi]).reshape((-1, 1))

    d = m_W.reshape((-1, 1)) - np.matmul(Rb_W, Oc_B) - Ob_W 
    m_C = np.matmul(Rc_B.T, np.matmul(Rb_W.T, d))

    ## divide m_C over its Z component.
    m_C = m_C / m_C[2]
    return m_C

def computerFeatureProjectionOnImagePlane(Px: Traj1D, Py: Traj1D, Pz: Traj1D, Pyaw: Traj1D, Rc_B, Oc_B, m):
    '''
        @arg m is the position of the visual feature in the world frame.
    '''
    xi = Px.dP_dt_list[0]
    yi = Py.dP_dt_list[0]
    zi = Pz.dP_dt_list[0]

    theta = Pyaw.dP_dt_list[0]

    xi_acc = Px.dP_dt_list[2]
    yi_acc = Py.dP_dt_list[2]
    zi_acc = Pz.dP_dt_list[2]

    W = ReferenceFrame('W')
    Ob_W = xi*W.x + yi*W.y + zi*W.z

    g = 9.8
    Zb_W = (xi_acc * W.x + yi_acc * W.y + (zi_acc + g) * W.z).normalize()

    Xc_W = sympy.cos(theta)*W.x + sympy.sin(theta)*W.y + 0*W.z

    Yb_W = Zb_W.cross(Xc_W).normalize()
    Xb_W = Yb_W.cross(Zb_W)

    ## create Rb_W
    M1 = Xb_W.to_matrix(W)
    M2 = Yb_W.to_matrix(W)
    M3 = Zb_W.to_matrix(W)

    Rb_W = M1.row_join(M2).row_join(M3)

    Ob_W = Matrix([xi, yi, zi])
    m_W = Matrix(m)
    Oc_B = Matrix(Oc_B)

    d = m_W - (Rb_W * Oc_B) - Ob_W 

    m_C = Rc_B.T * Rb_W.T * d
    m_C = m_C / m_C[2]
    return m_C

def add_visual_features_as_ineq_constraints(M, Px, Py, Pz, Pyaw, Rc_B, Oc_B, ti_list, constraints):
    for i in range(M.shape[0]):
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, M[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, M[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, M[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, M[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)

def objectiveFunction(x, alpha1, mc_fun, alpha2, mc_dot_mag_fun, ti_list):
    T = x[-1]
    ti_list = ti_list * T
    mc_dot_fun_value = mc_dot_mag_fun(x, ti_list)
    print('mc_dot', mc_dot_fun_value.shape)
    mc_dot_mean = mc_dot_fun_value.mean()
    
    mc_fun_value = np.array([mc_fun(x, ti)[:2] for ti in ti_list])
    mc_squared = np.square(mc_fun_value)
    mc_squared_mean = mc_squared.mean()
    print('mc_squared:', mc_squared.shape, 'mc_squared_mean', mc_squared_mean.shape)
    return T + alpha1 * mc_squared_mean + alpha2 + mc_dot_mean


def solve_n_D_OptimizationProblem(startPose=[-5, -30, 1, np.pi/3.], endPose=[0, -4, 2.5, np.pi/2.], \
        featuresWorldPosition=np.array([[0.0, 0.0, 4.03849]]), camera_FOV_h=80, camera_FOV_v=60):

    # n >= k-1, the number of control points is n+1
    # n + 1 >= k
    n_p = 16 # number of control points is (n+1)
    n_yaw = 6

    d_p = 4
    d_yaw = 2

    knots_p = createNormalizedClampKnotVector(n_p+1, d_p)
    knots_yaw = createNormalizedClampKnotVector(n_yaw+1, d_yaw)

    numT = 1
    Px = Traj1D(d_p, n_p, knots_p, numT)    
    Py = Traj1D(d_p, n_p, knots_p, numT)    
    Pz = Traj1D(d_p, n_p, knots_p, numT)    
    Pyaw = Traj1D(d_yaw, n_yaw, knots_yaw, numT)    

    ## visual features stuff:
    FOV_h = camera_FOV_h*np.pi/180.0
    FOV_v = camera_FOV_v*np.pi/180.0
    M = featuresWorldPosition
    Rc_B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    Oc_B = np.zeros((3, 1))


    # dP_dt_dC_list, dP_dt_dT_list = createGradList(dP_dt_list, T, C)
    # dP_dt_dC_lambdified_list = [lambdifyList(dp_list, t, T, C) for dp_list in dP_dt_dC_list]
    # dP_dt_dT_lambdified_list = lambdifyList(dP_dt_dT_list, t, T, C)

    x_len = Px.cpLen + Py.cpLen + Pz.cpLen + Pyaw.cpLen + numT

    Px.setControlPointsIndices(0, Px.cpLen)
    Py.setControlPointsIndices(Px.cpLen, Px.cpLen + Py.cpLen)
    Pz.setControlPointsIndices(Px.cpLen + Py.cpLen, Px.cpLen + Py.cpLen + Pz.cpLen)
    Pyaw.setControlPointsIndices(Px.cpLen + Py.cpLen + Pz.cpLen, Px.cpLen + Py.cpLen + Pz.cpLen + Pyaw.cpLen)

    initial_yaw = startPose[-1]

    x0 = create_initial_sol(numT, Px, Py, Pz, Pyaw, startPose, endPose)

    mc = computerFeatureProjectionOnImagePlane(Px, Py, Pz, Pyaw, Rc_B, Oc_B, M[0])

    print('diff of mc, started')
    ts = time.time()
    mc_dot = diff(mc, Traj1D.t)
    te = time.time()
    print('diff of mc, finished', te-ts)

    mc_dt_magSquared = mc_dot[0]**2 + mc_dot[1]**2

    x_args = Px.C + Py.C + Pz.C + Pyaw.C + [Traj1D.T]

    print('lambdifiying mc, started')
    ts = time.time()
    mc_fun = lambdify((x_args, Traj1D.t), mc, 'numpy')
    te = time.time()
    print('lambdifiying mc, finished', te-ts)

    print('lambdifiying mc_dot_mag...')
    ts = time.time()
    mc_dt_magsquard_fun = lambdify((x_args, Traj1D.t), mc_dt_magSquared, 'numpy')
    te = time.time()
    print('lambdifiying mc_dot_mag was finished', te-ts)

    ### adding equality constraints:
    ## inital conditions for the position and the reset of the derivatives:
    initialConditionList_Px = [0.0] * (d_p+1)
    initialConditionList_Px[0] = startPose[0] # inital position x(0) = 1
    initialConditionList_Py = initialConditionList_Px.copy()
    initialConditionList_Py[0] = startPose[1] # inital position x(0) = 1
    initialConditionList_Pz = initialConditionList_Px.copy()
    initialConditionList_Pz[0] = startPose[2] # inital position x(0) = 1

    initalConditionList_yaw = [0.0] * (d_yaw+1)
    initalConditionList_yaw[0] = initial_yaw # inital orientation yaw(0) = 45 degree

    constraints = []

    ## initial position constraints
    for i in range(d_p+1): # we have (d+1) conditions; the position and (d) derivatives.
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Px, i, initialConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Py, i, initialConditionList_Py[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Pz, i, initialConditionList_Pz[i])} 
        constraints.append(eq_cons)

    ## inital orientation constraints
    for i in range(d_yaw+1):
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Pyaw, i, initalConditionList_yaw[i])} 
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
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (Px, i, endConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (Py, i, endConditionList_Py[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (Pz, i, endConditionList_Pz[i])} 
        constraints.append(eq_cons)

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
    for i in range(1, d_p+1): # start from 1 so you skip the position
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Px, i, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Px, i, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Py, i, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Py, i, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Pz, i, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Pz, i, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
    
    for i in range(1, d_yaw+1): # start from 1 so you skip the heading
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Pyaw, i, orientationTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Pyaw, i, orientationTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
    
    # ## visual features constraints:
    # add_visual_features_as_ineq_constraints(M, Px, Py, Pz, Pyaw, Rc_B, Oc_B, ti_list, constraints)

    bounds = [(None, None)] * (x_len)
    bounds[-numT:] = [(0.1, None)] * numT


    skip_opt = False
    if not skip_opt:
        num = 50
        ti_list = np.linspace(0, 1, num, endpoint=True)
        alpha1 = 40.
        alpha2 = 10.
        res = optimize.minimize(fun=objectiveFunction, args=(alpha1, mc_fun, alpha2, mc_dt_magsquard_fun, ti_list), x0=x0, method='SLSQP', bounds=bounds, constraints=constraints,
                        tol=1e-6, options={'maxiter': 1e8, 'disp': True})
        print(res)
        x_star = res.x
    else:
        print('optimization skipped')

        x_star_took_years_to_get = np.array([-5.00000000e+00, -5.00000000e+00, -5.00000000e+00, -5.00000000e+00,
       -5.00000000e+00, -5.63952907e+00, -6.73693307e+00, -7.28937461e+00,
       -6.65582589e+00, -5.01644230e+00, -2.69118122e+00, -8.21183225e-01,
        2.00870295e-24,  9.68910518e-26, -6.38852533e-28, -3.26895809e-32,
       -1.00650880e-32, -3.00000000e+01, -3.00000000e+01, -3.00000000e+01,
       -3.00000000e+01, -3.00000000e+01, -2.91788168e+01, -2.66470419e+01,
       -2.23364503e+01, -1.70000000e+01, -1.16635497e+01, -7.35295806e+00,
       -4.82118322e+00, -4.00000000e+00, -4.00000000e+00, -4.00000000e+00,
       -4.00000000e+00, -4.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.74904865e+00,
        3.17501138e+00,  4.38457039e+00,  4.57140693e+00,  3.75038546e+00,
        2.75755369e+00,  2.35909262e+00,  2.50000000e+00,  2.50000000e+00,
        2.50000000e+00,  2.50000000e+00,  2.50000000e+00,  1.04719755e+00,
        1.04719755e+00,  1.04719755e+00,  1.33466254e+00,  1.17864133e+00,
        1.70273708e+00,  1.53889576e+00,  1.04062887e+01])


    T_star_list = x_star[-numT:]
    T_star = T_star_list.sum()
    print('T_star =', T_star)

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
