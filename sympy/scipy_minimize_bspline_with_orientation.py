from math import sin, cos
import numpy as np
from numpy import linalg as la
from scipy import interpolate
from scipy import optimize
import time

import matplotlib.pyplot as plt

from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, summation, Piecewise
from sympy.functions.special.bsplines import bspline_basis, _add_splines

from robot2D import isInFOV


class Traj1D:
    t = Symbol('t')
    T = Symbol('T')
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
        self.C = [Symbol('C{}'.format(i)) for i in range(self.cpLen)]
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
        lambdified_list.append(lambdify((t, T, C), f))
    return lambdified_list

def objectiveFunction(x, numT):
    return x[-numT:].sum()

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

def inFOV_ineq_constraint(x, Px:Traj1D, Py:Traj1D, Pyaw:Traj1D, FOV, m, ratio, sign):
    assert abs(sign) == 1
    _, T = Px.applyOptVect(x) 
    t = ratio * T
    xi = Px.diP_dt(x, 0, t)
    yi = Py.diP_dt(x, 0, t)
    theta = Pyaw.diP_dt(x, 0, t) 

    R = np.array([[cos(theta), sin(theta)],
                [-sin(theta), cos(theta)]])
    d = (m - np.array([xi, yi]) ).reshape((-1, 1))
    v = np.matmul(R, d).reshape((-1,))

    v_fov = la.norm(v) * sin(FOV/2)

    if sign == 1:
        g = v_fov - v[1]
    else:
        g = v[1] + v_fov
    return g
    
    


def solve_n_D_OptimizationProblem(startPose=(1, 1, 1, np.pi/4), endPose=(20, 10, 1, np.pi/4), featuresWorldPosition=np.array([(18.5, 18), (21.5, 17)]), camera_FOV=45):
    # n >= k-1, the number of control points is n+1
    # n + 1 >= k
    n_p = 13 # number of control points is (n+1)
    n_yaw = 6

    d_p = 4
    d_yaw = 2

    knots_p = createNormalizedClampKnotVector(n_p+1, d_p)
    knots_yaw = createNormalizedClampKnotVector(n_yaw+1, d_yaw)

    numT = 1
    Px = Traj1D(d_p, n_p, knots_p, numT)    
    Py = Traj1D(d_p, n_p, knots_p, numT)    
    Pyaw = Traj1D(d_yaw, n_yaw, knots_yaw, numT)    

    ## visual features stuff:
    FOV = camera_FOV*np.pi/180.0
    M = featuresWorldPosition[:, :2]


    # dP_dt_dC_list, dP_dt_dT_list = createGradList(dP_dt_list, T, C)
    # dP_dt_dC_lambdified_list = [lambdifyList(dp_list, t, T, C) for dp_list in dP_dt_dC_list]
    # dP_dt_dT_lambdified_list = lambdifyList(dP_dt_dT_list, t, T, C)

    x_len = Px.cpLen + Py.cpLen + Pyaw.cpLen + numT
    Px.setControlPointsIndices(0, Px.cpLen)
    Py.setControlPointsIndices(Px.cpLen, Px.cpLen + Py.cpLen)
    Pyaw.setControlPointsIndices(Px.cpLen + Py.cpLen, Px.cpLen + Py.cpLen + Pyaw.cpLen)

    initial_yaw = startPose[-1]

    x0_cpx = np.linspace(startPose[0], endPose[0], num=Px.cpLen, endpoint=True)
    x0_cpy = np.linspace(startPose[1], endPose[1], num=Py.cpLen, endpoint=True)

    x0 = np.zeros((x_len,))
    x0[0:Px.cpLen] = x0_cpx
    x0[Px.cpLen:Px.cpLen+Py.cpLen] = x0_cpy
    x0[Px.cpLen + Py.cpLen:Px.cpLen + Py.cpLen + Pyaw.cpLen] = initial_yaw
    T0_list = [50, 50, 50]
    for i in range(1, numT+1):
        x0[-i] = T0_list[-i] 

    print('x0 =', x0)

    # x_star = np.array([ -3.        ,  -3.        ,  -3.        ,  -3.        ,
    # -3.        ,  -2.6       ,  -1.64      ,  -0.36      ,
    #     0.6       ,   1.        ,   1.        ,   1.        ,
    #     1.        ,   1.        , -12.        , -12.        ,
    # -12.        , -12.        , -12.        , -12.        ,
    # -12.        , -12.        , -12.        , -11.99999992,
    # -11.99999993, -11.99999995, -11.99999996, -11.99999996,
    #     1.57079633,   1.57079633,   1.57079633,   1.57079633,
    #     1.57079633,   1.57079633,   1.57079633,   6.68740305]) 

    # return x_star, Px, Py, Pyaw

    
    ### adding equality constraints:
    ## inital conditions for the position and the reset of the derivatives:
    initialConditionList_Px = [0.0] * (d_p+1)
    initialConditionList_Px[0] = startPose[0] # inital position x(0) = 1
    initialConditionList_Py = initialConditionList_Px.copy()
    initialConditionList_Py[0] = startPose[1] # inital position x(0) = 1

    initalConditionList_yaw = [0.0] * (d_yaw+1)
    initalConditionList_yaw[0] = initial_yaw # inital orientation yaw(0) = 45 degree

    constraints = []

    ## initial position constraints
    for i in range(d_p+1): # we have (d+1) conditions; the position and (d) derivatives.
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Px, i, initialConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Py, i, initialConditionList_Py[i])} 
        constraints.append(eq_cons)

    ## inital orientation constraints
    for i in range(d_yaw+1):
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Pyaw, i, initalConditionList_yaw[i])} 
        constraints.append(eq_cons)

    
    ## end conditions for the position and the reset of the derivatives:
    ## there is no end conditions for the orientation
    endConditionList_Px = initialConditionList_Px.copy()
    endConditionList_Py = initialConditionList_Py.copy()
    endConditionList_Px[0] = endPose[0]
    endConditionList_Py[0] = endPose[1]

    for i in range(d_p+1): 
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (Px, i, endConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (Py, i, endConditionList_Py[i])} 
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
    
    for i in range(1, d_yaw+1): # start from 1 so you skip the heading
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Pyaw, i, orientationTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (Pyaw, i, orientationTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
    
    ## visual features constraints:
    for i in range(len(M)):
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pyaw, FOV, M[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pyaw, FOV, M[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)


    bounds = [(None, None)] * (x_len)
    bounds[-numT:] = [(0.1, None)] * numT

    ## the len of the optVars is (n+2) = (n+1 the num of coeff) + 1 (T)
    print('optimization started')
    res = optimize.minimize(fun=objectiveFunction, args=(numT), x0=x0, method='SLSQP', bounds=bounds, constraints=constraints,
                    tol=1e-6, options={'maxiter': 1e8})
    print(res)

    x_star = res.x
    T_star_list = x_star[-numT:]
    T_star = T_star_list.sum()
    print('T_star =', T_star)

    t_list = np.linspace(0, T_star, num=1000, endpoint=True)
    Px_values = Px.diP_dt(x_star, 0, t_list)
    Py_values = Py.diP_dt(x_star, 0, t_list)

    for i in range(Px.d+1):
        dPx_dt_max = Px.diP_dt(x_star, i, t_list).max()
        dPx_dt_min = Px.diP_dt(x_star, i, t_list).min()
        print('{}th Px: max={}, min={}'.format(i, dPx_dt_max, dPx_dt_min))
    for i in range(Px.d+1):
        dPy_dt_max = Py.diP_dt(x_star, i, t_list).max()
        dPy_dt_min = Py.diP_dt(x_star, i, t_list).min() 
        print('{}th Py: max={}, min={}'.format(i, dPy_dt_max, dPy_dt_min))

    mag = 20
    # plt.subplot(1, 3, 1)
    plt.plot(Px_values, Py_values, 'b:', label='minimue-time trajectory')
    for ti in np.linspace(0, T_star, num=15, endpoint=True):
        x = Px.diP_dt(x_star, 0, ti)
        y = Py.diP_dt(x_star, 0, ti)
        theta = Pyaw.diP_dt(x_star, 0, ti)
        dx1 = mag * cos(theta+FOV/2)
        dy1 = mag * sin(theta+FOV/2)
        dx2 = mag * cos(theta-FOV/2)
        dy2 = mag * sin(theta-FOV/2)
        color = 'g' 
        for m in M:
            if isInFOV((x, y, theta), FOV, m) == False:
                color = 'r'
        plt.arrow(x, y, dx1, dy1, color=color)
        plt.arrow(x, y, dx2, dy2, color=color)


    plt.plot(M[:, 0], M[:, 1], 'r*', label='tracked visual features')
    plt.plot(initialConditionList_Px[0], initialConditionList_Py[0], 'ro', label='initial position')
    plt.plot(endConditionList_Px[0], endConditionList_Py[0], 'go', label='end position')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    # plt.plot(p_mid1[0], p_mid1[1], '*')
    # plt.plot(p_mid2[0], p_mid2[1], '*')
    # plt.subplot(1, 3, 2)
    # plt.plot(t_list, Px_values)
    # plt.subplot(1, 3, 3)
    # plt.plot(t_list, Py_values)
    plt.show()

    return x_star, Px, Py, Pyaw





if __name__ == '__main__':
    solve_n_D_OptimizationProblem()
