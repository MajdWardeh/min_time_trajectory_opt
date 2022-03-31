import numpy as np
from scipy import interpolate
from scipy import optimize
import nlopt

import matplotlib.pyplot as plt


import sympy
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, summation, Piecewise
from sympy.functions.special.bsplines import bspline_basis, _add_splines


Px_lambdified = None
d1Px_dT_lambdified = None
d1Px_dCx_lambdified = None
diP_dt = None

Py_lambdified = None
d1Py_dT_lambdified = None
d1Py_dCy_lambdified = None
diPx_dt_lambdified = None
djdiPx_dt_dCx_lambdified = None

Cx_len = None
Cy_len = None

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
    Bd0 = bspline_basis(d, knots, 0, t)
    P = _add_splines(0, 0, C[0], Bd0)
    for i in range(1, cp_len):
        Bdi = bspline_basis(d, knots, i, t)
        P = _add_splines(1, P, C[i], Bdi)
    return P, t, T, C

def createSimplePolynomial(n=4):
    t = Symbol('t')
    T = Symbol('T')
    C = [Symbol('C{}'.format(i)) for i in range(n+1)]
    P = 0
    for i in range(n+1):
        P += C[i] * pow(t/T, i)
    return P, t, T, C


def createTimeDerivativeList(P, t, T, C):
    diP_dt_list = [P]
    for i in range(len(C)-1):
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


def testingSimplePolynomial():
    P, t, T, C = createSimplePolynomial()
    diP_dt_list = createTimeDerivativeList(P, t, T, C)
    for diP_dt in diP_dt_list:
        print(diP_dt)


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
    h = cond - f(t, T, C)
    return h


def solveOptimizationProblem():
    n = 4
    P, t, T, C = createSimplePolynomial(n)

    print('P:')
    print(P)
    print('------------------')

    dP_dt_list = createTimeDerivativeList(P, t, T, C)
    print('dP_dt_list')
    for dp in dP_dt_list:
        print(dp)
        print('--------')
    print('------------------')
        

    dP_dt_lambdified_List = lambdifyList(dP_dt_list, t, T, C)

    dP_dt_dC_list, dP_dt_dT_list = createGradList(dP_dt_list, T, C)

    dP_dt_dC_lambdified_list = [lambdifyList(dp_list, t, T, C) for dp_list in dP_dt_dC_list]
    dP_dt_dT_lambdified_list = lambdifyList(dP_dt_dT_list, t, T, C)

    x0 = np.random.rand(n+2)
    x0[-1] = 100

    ### adding equality constraints:
    ## inital conditions for the position and the reset of the derivatives:
    initalConditionList = [0.0] * (n+1)
    initalConditionList[0] = 1.0 # inital position x(0) = 1

    constraints = []

    # equalityConstraints = []
    for i in range(n+1): # we have (n+1) conditions; the position and (n) derivatives.
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (dP_dt_lambdified_List[i], initalConditionList[i], 0)} 
        constraints.append(eq_cons)
        # eq = lambda x: constraint(x, dP_dt_lambdified_List[i], initalConditionList[i], t=0)
        # print(eq(x0))
        # equalityConstraints.append(eq)
    
    ## end conditions for the position and the reset of the derivatives:
    endConditionList = initalConditionList.copy()
    endConditionList[0] = 15.0
    print('len(endConditionList)', len(endConditionList))
    print('len(dP_dt_lambdified_List)', len(dP_dt_lambdified_List))
    for i in range(n+1): # we have (n+1) conditions; the position and (n) derivatives.
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (dP_dt_lambdified_List[i], endConditionList[i])} 
        constraints.append(eq_cons)
        # eq = lambda x: constraint(x, dP_dt_lambdified_List[i], endConditionList[i], t=x[-1])
        # print(eq(x0))
        # equalityConstraints.append(eq)
                                    
    ### adding inequality constraints:
    inequalityConstraintsList = []
    timeDerivativesLimitList = [1 for i in range(n+1)] 
    timeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    num = 50
    ti_list = np.linspace(0, 1, num, endpoint=True)
    # for i in range(1, n+1): # start from 1 so you skip the position
    #     for k in range(num):
    #         ineq_cons = {'type': 'ineq', 'fun': constraint_ineq, 'args': (dP_dt_lambdified_List[i], timeDerivativesLimitList[i], ti_list[k])} 
    #         constraints.append(ineq_cons)
            # ineq = lambda x: constraint(x, dP_dt_lambdified_List[i], timeDerivativesLimitList[i], t=ti_list[k]*x[-1])
            # inequalityConstraintsList.append(ineq)

    print('x0', x0)
    print('len x0', len(x0))

    bounds = [(None, None)] * (n+2)
    bounds[-1] = (0, None)

    # for i, eq in enumerate(equalityConstraints):
    #     constraints.append({'type': 'eq', 'fun': eq})

    # for i, ineq in enumerate(inequalityConstraintsList):
    #     constraints.append({'type': 'ineq', 'fun': ineq})

    # print(equalityConstraints)
    # for eq in equalityConstraints:
    #     print(" ", eq(x0))
    
    
        
    ## the len of the optVars is (n+2) = (n+1 the num of coeff) + 1 (T)
    res = optimize.minimize(fun=objectiveFunction, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints)
    print(res)



    
    

def testing_sympyImplemetation():

    ctr =np.array( [(3 , 1), (2.5, 4), (0, 1), (-2.5, 4),
                    (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1),])
    x=ctr[:,0]
    y=ctr[:,1]

    l=len(x)  
    d = 3

    knots=np.linspace(0,1,l-2,endpoint=True)
    knots=np.append([0,0,0],knots)
    knots=np.append(knots,[1,1,1])

    T1 = 10
    knotsNormalized = knots
    knots = knots * T1

    t = Symbol('t')
    T = Symbol('T')
    # tau = t / T

    Px, Cx = createBSpline(t/T, d, knotsNormalized, l)

    # print('Px', Px)
    # print('------------------')

    d1Px_dT = diff(Px, T)
    # print('d1Px_dT', d1Px_dT)
    # print('------------------')

    d1Px_dCx = [diff(Px, Ci) for Ci in Cx]

    diPx_dt = [Px]
    for i in range(d): # it should be d (the same order of the Bspline)
        diPx_dt.append(diff(diPx_dt[-1], t))
    
    djdiPx_dt_dCx = []
    for dPxdt in diPx_dt:
        djdiPx_dt_dCx.append([diff(dPxdt, Ci) for Ci in Cx])
    
    djdiPx_dt_dT = [diff(dPxdt, T) for dPxdt in diPx_dt]


    # for i, d1Px_dCxi in enumerate(d1Px_dCx):
    #     print('d1Px_dC{}'.format(i))
    #     print(d1Px_dCxi)
    #     print('------------------')

    Cx = tuple(Cx)

    Px_lambdified = lambdify((t, T, Cx), Px)
    
    d1Px_dT_lambdified = lambdify((t, T, Cx), d1Px_dT)

    d1Px_dCx_lambdified = [lambdify( (t, T, Cx), d1Px_dCi) for d1Px_dCi in d1Px_dCx]

    diPx_dt_lambdified = [lambdify((t, T, Cx), diPx) for diPx in diPx_dt]

    djdiPx_dt_dCx_lambdified = []
    for j, ddPx_list in enumerate(djdiPx_dt_dCx):
        for ddPxi in ddPx_list:
            ddPx_lambdified = [lambdify((t, T, Cx), ddPxi)]


    C_test = tuple(x)

    return 

    u3=np.linspace(0,1*T1,(max(l*2,70)),endpoint=True)


    gd_res = interpolate.splev(u3, [knots, [x], d])[0]
    pred_res = []
    for i in range(u3.shape[0]):
        pred_res.append(Px_lambdified(u3[i], T1, C_test))
    print(gd_res)
    print(pred_res)

    tck=[knots, [x,y], d]
    out = interpolate.splev(u3,tck)

    plt.plot(x,y,'k--',label='Control polygon',marker='o',markerfacecolor='red')
    #plt.plot(x,y,'ro',label='Control points only')
    plt.plot(out[0],out[1],'b',linewidth=2.0,label='B-spline curve')
    plt.legend(loc='best')
    plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.title('Cubic B-spline curve evaluation')
    plt.show()


if __name__ == '__main__':
    # others()
    # test_B_order2_0()
    # testing_sympyImplemetation()
    # testingSimplePolynomial()
    solveOptimizationProblem()
