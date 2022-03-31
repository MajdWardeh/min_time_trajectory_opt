import numpy as np
from scipy import interpolate
from scipy import optimize
import nlopt

import matplotlib.pyplot as plt


import sympy
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, summation, Piecewise
from sympy.functions.special.bsplines import bspline_basis, _add_splines


def createNormalizedClampKnotVector(n, k):
    t=np.linspace(0,1,n-(k-1),endpoint=True)
    # t=np.linspace(0,1,n-k+2,endpoint=True)
    t=np.append([0]*k,t)
    t=np.append(t,[1]*k)
    return t

    
def createBSpline(d, knots, cp_len):
    t = Symbol('t')
    T = Symbol('T')
    knots = tuple(knots)
    C = [Symbol('C{}'.format(i)) for i in range(cp_len)]
    Bd0 = bspline_basis(d, knots, 0, t/T)
    P = _add_splines(0, 0, C[0], Bd0)
    for i in range(1, cp_len):
        Bdi = bspline_basis(d, knots, i, t/T)
        P = _add_splines(1, P, C[i], Bdi)
    return P, t, T, C


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



def objectiveFunction(x):
    return x[-1]

def start_constraint(x, f, cond):
    '''
        equality:
            f(0) = condition
            f(0) - condition = 0
            h(optVars) = 0
        inequality:
            f(ti) <= cond
            f(ti) - cond <= 0
            cond - f(ti) >= 0 (scipy form)
            ineq(optVars) <= 0
    '''
    T = x[-1]
    C = x[0:-1]
    t = 0.0
    h = cond - f(t, T, C)
    return h

def end_constraint(x, f, cond):
    '''
        equality:
            f(0) = condition
            f(0) - condition = 0
            h(optVars) = 0
        inequality:
            f(ti) <= cond
            f(ti) - cond <= 0
            cond - f(ti) >= 0 (scipy form)
            ineq(optVars) <= 0
    '''
    t = x[-1]
    T = x[-1]
    C = x[0:-1]
    h = cond - f(t, T, C)
    return h

def constraint_ineq(x, f, cond, ratio):
    T = x[-1]
    C = x[0:-1]
    t = ratio * T
    g = cond - f(t, T, C)
    return g


def solveOptimizationProblem():
    # n >= k-1, the number of control points is n+1
    # n + 1 >= k
    n = 10 # number of control points is (n+1)
    d = 4
    knots = createNormalizedClampKnotVector(n+1, d)
    
    P, t, T, C = createBSpline(d, knots, n+1)

    dP_dt_list = createTimeDerivativeList(P, t, T, C)
    dP_dt_lambdified_List = lambdifyList(dP_dt_list, t, T, C)

    dP_dt_dC_list, dP_dt_dT_list = createGradList(dP_dt_list, T, C)

    dP_dt_dC_lambdified_list = [lambdifyList(dp_list, t, T, C) for dp_list in dP_dt_dC_list]
    dP_dt_dT_lambdified_list = lambdifyList(dP_dt_dT_list, t, T, C)

    x0 = np.random.rand(n+2)
    x0[-1] = 100


    ### adding equality constraints:
    ## inital conditions for the position and the reset of the derivatives:
    initalConditionList = [0.0] * (d+1)
    initalConditionList[0] = 1.0 # inital position x(0) = 1

    constraints = []

    # equalityConstraints = []
    for i in range(d+1): # we have (n+1) conditions; the position and (n) derivatives.
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (dP_dt_lambdified_List[i], initalConditionList[i])} 
        constraints.append(eq_cons)
    
    ## end conditions for the position and the reset of the derivatives:
    endConditionList = initalConditionList.copy()
    endConditionList[0] = 15.0

    for i in range(d+1): # we have (n+1) conditions; the position and (n) derivatives.
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (dP_dt_lambdified_List[i], endConditionList[i])} 
        constraints.append(eq_cons)
                                    
    ### adding inequality constraints:
    inequalityConstraintsList = []
    timeDerivativesLimitList = [2 for i in range(d+1)] 
    timeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    timeDerivativesLimitList[1] = 10 # vel
    timeDerivativesLimitList[2] = 5 # acc

    num = 20
    ti_list = np.linspace(0, 1, num, endpoint=True)
    for i in range(1, d+1): # start from 1 so you skip the position
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': constraint_ineq, 'args': (dP_dt_lambdified_List[i], timeDerivativesLimitList[i], ti_list[k])} 
            constraints.append(ineq_cons)


    bounds = [(None, None)] * (n+2)
    bounds[-1] = (0.5, None)

        
    ## the len of the optVars is (n+2) = (n+1 the num of coeff) + 1 (T)
    res = optimize.minimize(fun=objectiveFunction, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints,
                    tol=1e-6, options={'maxiter': 1e8})
    print(res)

    ## ploting the resutls:
    C_star = res.x[:-1]
    T_star = res.x[-1]
    P_lambdified = dP_dt_lambdified_List[0]
    t_list = np.linspace(0, T_star, num=1000, endpoint=True)
    P0_list1 = P_lambdified(t_list, T_star, C_star)

    for i in range(len(dP_dt_lambdified_List)):
        dP_dt_max = dP_dt_lambdified_List[i](t_list, T_star, C_star).max()
        print('max {}th P:'.format(i), dP_dt_max)

    plt.plot(t_list, P0_list1)
    plt.show()



    
    



if __name__ == '__main__':
    solveOptimizationProblem()
