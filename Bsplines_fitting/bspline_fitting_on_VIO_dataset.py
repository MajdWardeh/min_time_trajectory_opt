import os
from os import sys, path
sys.path.append('/home/majd/papers/Python-B-spline-examples')
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import cvxpy as cp
import time

from scipy_minimize_bspline_with_orientation import Traj1D, createNormalizedClampKnotVector 

base_dir = './Bsplines_fitting'

def getData():
    file1 = open(os.path.join(base_dir, 'groundtruth.txt'), 'r')
    Lines = file1.readlines()
    data = []
    head = Lines[0]
    # print("data head: {}".format(head))
    for line in Lines[1:]:
        line_data = line.split(' ')
        data.append(line_data)
    file1.close()
    data = np.array(data, dtype=np.float64)
    return data

def objectiveFunction(x, Px: Traj1D, Py: Traj1D, Pz: Traj1D, t_list, T, x_list, y_list, z_list):
    sum = 0
    for i, ti in enumerate(t_list):
        sum += (Px.diP_dt(x, 0, ti, T) - x_list[i])**2
    #     sum += (Py.diP_dt(x, 0, ti, T) - y_list[i])**2
    #     sum += (Pz.diP_dt(x, 0, ti, T) - z_list[i])**2
    return sum


def solve():
    ## data preparation
    data = getData()
    t_list = data[:, 1] - data[0, 1]
    T = 1
    x_list = data[:, 2] - data[0, 2]
    y_list = data[:, 3] - data[0, 3]
    z_list = data[:, 4] - data[0, 4]

    n_p = 13 # number of control points is (n+1)
    n_yaw = 6
    d_p = 4
    d_yaw = 2

    knots_p = createNormalizedClampKnotVector(n_p+1, d_p)
    knots_p = knots_p * T
    # knots_yaw = createNormalizedClampKnotVector(n_yaw+1, d_yaw)

    numT = 1
    Px = Traj1D(d_p, n_p, knots_p, numT)    
    Py = Traj1D(d_p, n_p, knots_p, numT)    
    Pz = Traj1D(d_p, n_p, knots_p, numT)    
    # Pyaw = Traj1D(d_yaw, n_yaw, knots_yaw, numT)    

    # x_len = Px.cpLen + Py.cpLen + Pz.cpLen
    x_len = Px.cpLen
    Px.setControlPointsIndices(0, Px.cpLen)
    Py.setControlPointsIndices(Px.cpLen, Px.cpLen + Py.cpLen)
    Pz.setControlPointsIndices(Px.cpLen + Py.cpLen, Px.cpLen + Py.cpLen + Pz.cpLen)

    x0 = np.random.rand(x_len)

    constraints = []
    # ## initial position constraints
    # for i in range(d_p+1): # we have (d+1) conditions; the position and (d) derivatives.
    #     eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (Px, i, initialConditionList_Px[i])} 


    bounds = [(None, None)] * (x_len)
    objectiveFunctionArgs = (Px, Py, Pz, t_list, T, x_list, y_list, z_list)

    # objFun = objectiveFunction(x0, Px, Py, Pz, t_list, T, x_list, y_list, z_list)

    print('optimization started')
    res = optimize.least_squares(fun=objectiveFunction, args=objectiveFunctionArgs, x0=x0)
    print(res)


def main():
    solve()
    
if __name__ == "__main__":
    main()
