from casadi import *

import numpy as np
from scipy.interpolate import BSpline
# from scipy_minimize_bspline_3D_with_orientation_motionBlur import createNormalizedClampKnotVector


def createNormalizedClampKnotVector(n, k):
    t=np.linspace(0,1,n-(k-1),endpoint=True)
    # t=np.linspace(0,1,n-k+2,endpoint=True)
    t=np.append([0]*k,t)
    t=np.append(t,[1]*k)
    return t

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
        # self.Cyaw = MX.sym('Cyaw', self.n_yaw+1)

        self.x = vertcat(self.Cx, self.Cy, self.Cz, self.T) #self.Cyaw, self.T)
        self.cpXlen = (self.n_p + 1) 
        self.cpYlen = (self.n_p + 1) 
        self.cpZlen = (self.n_p + 1) 
        # self.cpYawlen = self.n_yaw + 1

        self.xlen = (self.n_p + 1) * 3 + 1 #(self.n_yaw + 1) + 1
        

        knots_p = createNormalizedClampKnotVector(self.n_p+1, self.d_p).tolist()
        # knots_yaw = createNormalizedClampKnotVector(self.n_yaw+1, self.d_yaw).tolist()

        dimention = 1
        self.Px = bspline(self.t/self.T, self.Cx, [knots_p], [self.d_p], dimention, {})
        self.Py = bspline(self.t/self.T, self.Cy, [knots_p], [self.d_p], dimention, {})
        self.Pz = bspline(self.t/self.T, self.Cz, [knots_p], [self.d_p], dimention, {})
        # self.Pyaw = bspline(self.t/self.T, self.Cyaw, [knots_yaw], [self.d_yaw], dimention, {})

        ## computing position time gradients
        self.Px_dt_list = [self.Px]
        self.Py_dt_list = [self.Py]
        self.Pz_dt_list = [self.Pz]
        for _ in range(self.d_p):
            self.Px_dt_list.append(gradient(self.Px_dt_list[-1], self.t))
            self.Py_dt_list.append(gradient(self.Py_dt_list[-1], self.t))
            self.Pz_dt_list.append(gradient(self.Pz_dt_list[-1], self.t))

        # ## computing heading time gradients
        # self.Pyaw_dt_list = [self.Pyaw]
        # for _ in range(self.d_yaw):
        #     self.Pyaw_dt_list.append(gradient(self.Pyaw_dt_list[-1], self.t))
 

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
        # Pyaw_dt_fun_list = []
        # for j in range(self.d_yaw+1):
        #     fun = Function('Pyaw_d{}t_fun'.format(j), [self.t, self.x], [self.Pyaw_dt_list[j]])
        #     Pyaw_dt_fun_list.append(fun)
        # self.P_dt_fun['Yaw'] = Pyaw_dt_fun_list



            

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


def solve_n_D_OptimizationProblem(startPose=[-5, -30, 1, np.pi/3.], endPose=[0, -4, 2.5, np.pi/2.], featuresWorldPosition=np.array([[9.014978e-01, 1.497865e-03, 2.961498],[-8.985040e-01, 1.497627e-03, 2.961498]]), camera_FOV_h=80, camera_FOV_v=60):
    # n >= k-1, the number of control points is n+1
    # n + 1 >= k
    n_p = 11 # number of control points is (n+1)
    n_yaw = 6

    d_p = 4
    d_yaw = 2



    numT = 1

    traj = Trajectory(n_p, n_yaw, d_p, d_yaw)

    ## visual features stuff:
    FOV_h = camera_FOV_h*np.pi/180.0
    FOV_v = camera_FOV_v*np.pi/180.0
    M = featuresWorldPosition
    Rc_B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    Oc_B = np.zeros((3, 1))


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

    g_list = []
    g_ub_list = []
    g_lb_list = []

    print('testing here')

    ### adding inequality constraints:
    positionTimeDerivativesLimitList = [2 for i in range(d_p+1)] 
    positionTimeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    positionTimeDerivativesLimitList[1] = 10 # vel
    positionTimeDerivativesLimitList[2] = 5 # acc

    orientationTimeDerivativesLimitList = [2 for i in range(d_p+1)] 
    orientationTimeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    orientationTimeDerivativesLimitList[1] = 10 # heading angualr vel limit rad/s 
    orientationTimeDerivativesLimitList[2] = 2 # heading angualr acc limit rad/s**2


    ## initial position constraints
    for i in range(d_p+1): # we have (d+1) conditions; the position and (d) derivatives.
        g = traj.P_dt_fun['X'][i](0.0001, traj.T, traj.Cx) - initialConditionList_Px[i]
        g_list.append(g)
        g_ub_list.append(0.0)
        g_lb_list.append(0.0)
        # g = traj.P_dt_fun['Y'][i](0.0, traj.T, traj.Cy) - initialConditionList_Py[i]
        # g_list.append(g)
        # g_ub_list.append(0.0)
        # g_lb_list.append(0.0)
        # g = traj.P_dt_fun['Z'][i](0.0, traj.T, traj.Cz) - initialConditionList_Pz[i]
        # g_list.append(g)
        # g_ub_list.append(0.0)
        # g_lb_list.append(0.0)
    
    
    # ## inital orientation constraints
    # for i in range(d_yaw+1):
    #     g = traj.P_dt_fun['Yaw'][i](0.0, traj.x) - initalConditionList_yaw[i]
    #     g_list.append(g)
    #     g_ub_list.append(0.0)
    #     g_lb_list.append(0.0)

    num = 20
    ti_list = np.linspace(0, 1, num, endpoint=True)
    opsilon = 0.05
    for i in range(1, d_p+1): # start from 1 so you skip the position
        for ti in ti_list:
            if ti == 0:
                ti = 0.0001
            g = traj.P_dt_fun['X'][i](ti*traj.T, traj.T, traj.Cx)
            g_list.append(g)
            ## test using opsilon to make the equality inequality
            g_ub_list.append(positionTimeDerivativesLimitList[i] * (1 + opsilon))
            g_lb_list.append(-1 * positionTimeDerivativesLimitList[i] * (1 - opsilon))

    
    ## end conditions for the position and the reset of the derivatives:
    ## there is no end conditions for the orientation
    endConditionList_Px = initialConditionList_Px.copy()
    endConditionList_Py = initialConditionList_Py.copy()
    endConditionList_Pz = initialConditionList_Pz.copy()
    endConditionList_Px[0] = endPose[0]
    endConditionList_Py[0] = endPose[1]
    endConditionList_Pz[0] = endPose[2]

    for i in range(d_p+1): 
        g = traj.P_dt_fun['X'][i](traj.T, traj.T, traj.Cx) - endConditionList_Px[i]
        g_list.append(g)
        ## test using opsilon to make the equality inequality
        g_ub_list.append(opsilon)
        g_lb_list.append(-opsilon)
        # g = traj.P_dt_fun['Y'][i](traj.T, traj.T, traj.Cy) - endConditionList_Py[i]
        # g_list.append(g)
        # g_ub_list.append(0.0)
        # g_lb_list.append(0.0)
        # g = traj.P_dt_fun['Z'][i](traj.T, traj.T, traj.Cz) - endConditionList_Pz[i]
        # g_list.append(g)
        # g_ub_list.append(0.0)
        # g_lb_list.append(0.0)

                                    
                            

            # g = traj.P_dt_fun['Y'][i](ti*traj.T, traj.T, traj.Cy)
            # g_list.append(g)
            # g_ub_list.append(positionTimeDerivativesLimitList[i])
            # g_lb_list.append(-1 * positionTimeDerivativesLimitList[i])

            # g = traj.P_dt_fun['Z'][i](ti*traj.T, traj.T, traj.Cz)
            # g_list.append(g)
            # g_ub_list.append(positionTimeDerivativesLimitList[i])
            # g_lb_list.append(-1 * positionTimeDerivativesLimitList[i])
    
    # for i in range(1, d_yaw+1): # start from 1 so you skip the heading
    #     for ti in ti_list:
    #         g = traj.P_dt_fun['Yaw'][i](ti*traj.T, traj.x)
    #         g_list.append(g)
    #         g_ub_list.append(orientationTimeDerivativesLimitList[i])
    #         g_lb_list.append(-1 * orientationTimeDerivativesLimitList[i])
    
    # # ## visual features constraints:
    # for i in range(M.shape[0]):
    #     for k in range(len(ti_list)):
    #         ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, M[i], ti_list[k], 1)} 
    #         g_list.append(ineq_cons)
    #         ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 0, FOV_h, M[i], ti_list[k], -1)} 
    #         g_list.append(ineq_cons)
    #         ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, M[i], ti_list[k], 1)} 
    #         g_list.append(ineq_cons)
    #         ineq_cons = {'type': 'ineq', 'fun': inFOV_ineq_constraint, 'args': (Px, Py, Pz, Pyaw, Rc_B, Oc_B, 1, FOV_v, M[i], ti_list[k], -1)} 
    #         g_list.append(ineq_cons)

    lbx = [-inf] * (traj.cpXlen + 1)
    lbx[-1] = 0.5
    ubx = [inf] * (traj.cpXlen + 1)
    ubx[-1] = 150

    # Initial guess for the decision variables
    x0_cpx = np.linspace(startPose[0], endPose[0], num=traj.cpXlen, endpoint=True)
    x0_cpy = np.linspace(startPose[1], endPose[1], num=traj.cpYlen, endpoint=True)
    x0_cpz = np.linspace(startPose[2], endPose[2], num=traj.cpZlen, endpoint=True)
    # x0_cpyaw = np.linspace(startPose[3], endPose[3], num=traj.cpYawlen, endpoint=True)

    x0 = [0.0] * (traj.cpXlen + 1)
    x0[0:traj.cpXlen] = np.random.rand(traj.cpXlen)
    # x0[traj.cpXlen:traj.cpXlen + traj.cpYlen] = x0_cpy
    # x0[traj.cpXlen + traj.cpYlen:traj.cpXlen + traj.cpYlen + traj.cpZlen] = x0_cpz
    # x0[traj.cpXlen + traj.cpYlen + traj.cpZlen:traj.cpXlen + traj.cpYlen + traj.cpZlen + traj.cpYawlen] = x0_cpyaw
    T0_list = [50, 50, 50]
    for i in range(1, numT+1):
        x0[-i] = T0_list[-i] 


    ## Create NLP solver
    g = vcat(g_list)
    x = vertcat(traj.Cx, traj.T)
    nlp = {'x': x, 'f':traj.T, 'g':g}
    solver = nlpsol('S', 'ipopt', nlp)
    print(solver)

    sol = solver(x0=x0, lbg=g_lb_list, ubg=g_ub_list, lbx=lbx, ubx=ubx)
    print(sol['x'])
    print(x0)




    return 

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




if __name__=='__main__':
    solve_n_D_OptimizationProblem()