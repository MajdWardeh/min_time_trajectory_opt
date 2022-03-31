from matplotlib.pyplot import axis
import numpy as np
import numpy.linalg as la
from scipy.interpolate import BSpline
from scipy import optimize
# from scipy_minimize_bspline_3D_with_orientation_motionBlur import createNormalizedClampKnotVector


def createNormalizedClampKnotVector(n, k):
    t=np.linspace(0,1,n-(k-1),endpoint=True)
    # t=np.linspace(0,1,n-k+2,endpoint=True)
    t=np.append([0]*k,t)
    t=np.append(t,[1]*k)
    return t

class Trajectory:

    def __init__(self, np=12, nyaw=4, dp=4, dyaw=2, numT=1):
        self.n_p = np # number of control points is (n+1)
        self.n_yaw = nyaw
        self.d_p = dp
        self.d_yaw = dyaw
        self.numT = numT

        self.c_p_len = self.n_p + 1 
        self.c_h_len = self.n_yaw + 1

        c_p_opt_len = self.c_p_len - 2 * (self.d_p + 1)
        c_h_opt_len = self.c_h_len - 2 * (self.d_yaw + 1)
        self.c_pxij = (0, c_p_opt_len)
        self.c_pyij = (c_p_opt_len, c_p_opt_len*2)
        self.c_pzij = (c_p_opt_len*2, c_p_opt_len*3)
        self.c_hij = (c_p_opt_len*3, c_p_opt_len*3+c_h_opt_len)

        ## opt vars indices in sxe (Start X End)
        self.sxe_pxyz_ij = (self.d_p+1, self.d_p+1+c_p_opt_len)
        self.sxe_h_ij = (self.d_yaw+1, self.d_yaw+1+c_h_opt_len)

        self.xlen = (c_p_opt_len) * 3 + (c_h_opt_len) + self.numT

        self.knots_p = createNormalizedClampKnotVector(self.n_p+1, self.d_p)
        self.knots_h = createNormalizedClampKnotVector(self.n_yaw+1, self.d_yaw)
    
    def setStartEndConditions(self, startPose, endPose):
        '''
            adds start and end constraints to the trajectory, to reduce the opt variables.
        '''
        assert len(startPose) == 4
        assert len(endPose) == 4

        ## sxe: stands for Start X End
        self.c_p_sxe = np.zeros((3, self.c_p_len)) 
        for i in range(3):
            for j in range(self.d_p + 1):
                self.c_p_sxe[i, j] = startPose[i]
                self.c_p_sxe[i, -j-1] = endPose[i]
        self.c_h_sxe = np.zeros((self.c_h_len,)) 
        for i in range(self.d_yaw + 1):
            self.c_h_sxe[i] = startPose[-1]
            self.c_h_sxe[-i-1] = endPose[-1]
        print(self.c_p_sxe)
        print(self.c_h_sxe)

    def diP_dt(self, x, i, t, axis=-1):
        ''' 
            evaluate the value of ith derivative of a bspline @ x, t
            args:
                x: opt vector
                t: time var
                i: the ith derivative of the bspline
        '''
        T = x[-self.numT:].sum()
        if axis == -1:
            cpx_sxe = self.c_p_sxe[0, :]
            cpx_sxe[self.sxe_pxyz_ij[0]:self.sxe_pxyz_ij[1]] = x[self.c_pxij[0]:self.c_pxij[1]]

            cpy_sxe = self.c_p_sxe[1, :]
            cpy_sxe[self.sxe_pxyz_ij[0]:self.sxe_pxyz_ij[1]] = x[self.c_pyij[0]:self.c_pyij[1]]

            cpz_sxe = self.c_p_sxe[2, :]
            cpz_sxe[self.sxe_pxyz_ij[0]:self.sxe_pxyz_ij[1]] = x[self.c_pzij[0]:self.c_pzij[1]]

            ch_sxe = self.c_h_sxe[:]
            ch_sxe[self.sxe_h_ij[0]:self.sxe_h_ij[1]] = x[self.c_hij[0]:self.c_hij[1]]


            # c_px = x[self.c_pxij[0]:self.c_pxij[1]] 
            # c_py = x[self.c_pyij[0]:self.c_pyij[1]] 
            # c_pz = x[self.c_pzij[0]:self.c_pzij[1]] 
            # c_h = x[self.c_hij[0]:self.c_hij[1]] 
            Px = BSpline(self.knots_p, cpx_sxe, self.d_p)
            Py = BSpline(self.knots_p, cpy_sxe, self.d_p)
            Pz = BSpline(self.knots_p, cpz_sxe, self.d_p)
            Ph = BSpline(self.knots_h, ch_sxe, self.d_yaw)
            return np.array([Px(t/T, nu=i), Py(t/T, nu=i), Pz(t/T, nu=i), Ph(t/T, nu=i)]) / T**i
        elif axis == 0:
            cpx_sxe = self.c_p_sxe[0, :]
            cpx_sxe[self.sxe_pxyz_ij[0]:self.sxe_pxyz_ij[1]] = x[self.c_pxij[0]:self.c_pxij[1]]
            Px = BSpline(self.knots_p, cpx_sxe, self.d_p)
            return Px(t/T, nu=i) / T**i
        elif axis == 1:
            cpy_sxe = self.c_p_sxe[1, :]
            cpy_sxe[self.sxe_pxyz_ij[0]:self.sxe_pxyz_ij[1]] = x[self.c_pyij[0]:self.c_pyij[1]]
            Py = BSpline(self.knots_p, cpy_sxe, self.d_p)
            return Py(t/T, nu=i) / T**i
        elif axis == 2:
            cpz_sxe = self.c_p_sxe[2, :]
            cpz_sxe[self.sxe_pxyz_ij[0]:self.sxe_pxyz_ij[1]] = x[self.c_pzij[0]:self.c_pzij[1]]
            Pz = BSpline(self.knots_p, cpz_sxe, self.d_p)
            return Pz(t/T, nu=i) / T**i
        elif axis == 3:
            ch_sxe = self.c_h_sxe[:]
            ch_sxe[self.sxe_h_ij[0]:self.sxe_h_ij[1]] = x[self.c_hij[0]:self.c_hij[1]]
            Ph = BSpline(self.knots_h, ch_sxe, self.d_yaw) 
            return Ph(t/T, nu=i) / T**i
        else:
            raise ValueError


def create_initial_sol(P, startPose, endPose):
    x_len = P.xlen
    x0_cpx = np.linspace(startPose[0], endPose[0], num=P.c_p_len, endpoint=True)
    x0_cpy = np.linspace(startPose[1], endPose[1], num=P.c_p_len, endpoint=True)
    x0_cpz = np.linspace(startPose[2], endPose[2], num=P.c_p_len, endpoint=True)
    x0_cpyaw = np.linspace(startPose[3], endPose[3], num=P.c_h_len, endpoint=True)

    x0 = np.zeros((x_len,))
    x0[P.c_pxij[0]: P.c_pxij[1]] = x0_cpx
    x0[P.c_pyij[0]: P.c_pyij[1]] = x0_cpy
    x0[P.c_pzij[0]: P.c_pzij[1]] = x0_cpz
    x0[P.c_hij[0]: P.c_hij[1]] = x0_cpyaw

    x0[-P.numT:] = [50.] * P.numT 
    return x0

def start_constraint(x, P, i, axis, cond):
    '''
        equality constraint:
        diP_dt(0) = cond
        diP_dt(0) - cond = 0
    '''
    return cond - P.diP_dt(x, i, 0.0, axis)

def end_constraint(x, P, i, axis, cond):
    '''
        equality constraint:
        diP_dt(t[j]) = cond
        diP_dt(t[j]) - cond = 0
    '''
    T = x[-P.numT:].sum()
    return cond - P.diP_dt(x, i, T, axis)

def ineq_constraint(x, P, i, axis, cond, ratio, sign):
    '''
        sign * diP_dt(ratio*T) <= cond
        cond - sign * diP_dt(ratio*T) >= 0
        g >= 0
    '''
    assert abs(sign) == 1
    T = x[-1]
    t = ratio * T
    g = cond - sign * P.diP_dt(x, i, t, axis)
    return g

def compute_mC(x, P, ti_list, Rc_B, Oc_B, m, camera_k0):
    '''
        t: the time at which Rb_W is computed. t is in [0, T]
    '''
    Px_ti_list = P.diP_dt(x, 0, ti_list)
    xi = Px_ti_list[0]
    yi = Px_ti_list[1]
    zi = Px_ti_list[2]
    theta = Px_ti_list[3]

    xi_acc = P.diP_dt(x, 2, ti_list, axis=0)
    yi_acc = P.diP_dt(x, 2, ti_list, axis=1)
    zi_acc = P.diP_dt(x, 2, ti_list, axis=2)
    g = 9.806

    ZB = np.vstack([xi_acc, yi_acc, zi_acc+g])
    ZB = ZB / la.norm(ZB, axis=0)

    XC = np.vstack([np.cos(theta), np.sin(theta), np.zeros(theta.shape[0],)])

    YB = np.cross(ZB, XC, axis=0)
    YB = YB / la.norm(YB, axis=0)

    XB = np.cross(YB, ZB, axis=0)

    ## Rb_W is the rotation matrix from the body frame B to the world frame W.
    Rb_W = np.vstack([XB, YB, ZB])
    Rw_B = Rb_W.T
    Rw_B = Rw_B.reshape((-1, 3, 3))

    ## m_C = [Rc_B]^T * [Rb_W]^T * [m_w - (Rb_W * Oc_B) - Ob_w]
    Ob_W = np.array([xi, yi, zi]).T
    Ob_W = np.expand_dims(Ob_W, axis=-1)

    d = m.reshape(3, 1) - Ob_W 

    m_C = np.matmul(Rc_B.T, np.matmul(Rw_B, d))

    mI = np.matmul(camera_k0, m_C)
    mI2 = mI[:, 2]
    mI2 = np.expand_dims(mI2, axis=-1)
    mI = mI / mI2
    mI = mI[:, :2, :]
    mI = np.square(mI)
    loss_mI = mI.mean()
    return loss_mI, m_C, Rw_B, Ob_W

def objectiveFunction(x, P, ti_list, Rc_B, Oc_B, m, camera_k0, alpha1):
    T = x[-P.numT:].sum()
    ti_list = ti_list * T
    mI_loss, mC, Rw_B, Ob_W = compute_mC(x, P, ti_list, Rc_B, Oc_B, m, camera_k0)
    return T + alpha1 * mI_loss  

def solve_n_D_OptimizationProblem(startPose=[-5, -30, 1, np.pi/3.], endPose=[0, -4, 2.5, np.pi/2.], \
        featuresWorldPosition=np.array([[0.0, 0.0, 2.03849]]), camera_FOV_h=80, camera_FOV_v=60, skip_opt=False):

    n_p = 20 # number of control points is (n+1)
    n_yaw = 6
    d_p = 5
    d_yaw = 2

    P = Trajectory(n_p, n_yaw, d_p, d_yaw)
    P.setStartEndConditions(startPose, endPose)
    return

    numT = 1

    ## visual features stuff:
    FOV_h = camera_FOV_h*np.pi/180.0
    FOV_v = camera_FOV_v*np.pi/180.0
    M = featuresWorldPosition
    # camera_k = np.array([342.7555236816406, 0.0, 320.0, 0.0, 342.7555236816406, 240.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    camera_k0 = np.diag([10, 10, 1])
    Rc_B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    Oc_B = np.zeros((3, 1))


   ### adding equality constraints:
    ## inital conditions for the position and the reset of the derivatives:
    initialConditionList_Px = [0.0] * (d_p+1)
    initialConditionList_Px[0] = startPose[0] # inital position x(0) = 1
    initialConditionList_Py = initialConditionList_Px[:]
    initialConditionList_Py[0] = startPose[1] # inital position x(0) = 1
    initialConditionList_Pz = initialConditionList_Px[:]
    initialConditionList_Pz[0] = startPose[2] # inital position x(0) = 1

    initalConditionList_yaw = [0.0] * (d_yaw+1)
    initalConditionList_yaw[0] = startPose[-1] # inital orientation yaw(0) = 45 degree

    constraints = []

    ## initial position constraints
    for i in range(d_p+1): # we have (d+1) conditions; the position and (d) derivatives.
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (P, i, 0, initialConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (P, i, 1, initialConditionList_Py[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (P, i, 2, initialConditionList_Pz[i])} 
        constraints.append(eq_cons)

    ## inital orientation constraints
    for i in range(d_yaw+1):
        eq_cons = {'type': 'eq', 'fun': start_constraint, 'args': (P, i, 3, initalConditionList_yaw[i])} 
        constraints.append(eq_cons)

    
    ## end conditions for the position and the reset of the derivatives:
    ## there is no end conditions for the orientation
    endConditionList_Px = initialConditionList_Px[:]
    endConditionList_Py = initialConditionList_Py[:]
    endConditionList_Pz = initialConditionList_Pz[:]
    endConditionList_Px[0] = endPose[0]
    endConditionList_Py[0] = endPose[1]
    endConditionList_Pz[0] = endPose[2]

    for i in range(d_p+1): 
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (P, i, 0, endConditionList_Px[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (P, i, 1, endConditionList_Py[i])} 
        constraints.append(eq_cons)
        eq_cons = {'type': 'eq', 'fun': end_constraint, 'args': (P, i, 2, endConditionList_Pz[i])} 
        constraints.append(eq_cons)

    ### adding inequality constraints:
    positionTimeDerivativesLimitList = [6 for i in range(d_p+1)] 
    positionTimeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    positionTimeDerivativesLimitList[1] = 60 # vel
    positionTimeDerivativesLimitList[2] = 20 # acc

    orientationTimeDerivativesLimitList = [2 for i in range(d_p+1)] 
    orientationTimeDerivativesLimitList[0] = None # to make sure that the position is skipped.
    orientationTimeDerivativesLimitList[1] = 10 # heading angualr vel limit rad/s 
    orientationTimeDerivativesLimitList[2] = 2 # heading angualr acc limit rad/s**2

    num = 20
    ti_list = np.linspace(0, 1, num, endpoint=True)
    for i in range(1, d_p+1): # start from 1 so you skip the position
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 0, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 0, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 1, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 1, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 2, positionTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 2, positionTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)
    
    for i in range(1, d_yaw+1): # start from 1 so you skip the heading
        for k in range(len(ti_list)):
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 3, orientationTimeDerivativesLimitList[i], ti_list[k], 1)} 
            constraints.append(ineq_cons)
            ineq_cons = {'type': 'ineq', 'fun': ineq_constraint, 'args': (P, i, 3, orientationTimeDerivativesLimitList[i], ti_list[k], -1)} 
            constraints.append(ineq_cons)

    bounds = [(None, None)] * (P.xlen)
    bounds[-numT:] = [(0.1, None)] * numT

    x0 = create_initial_sol(P, startPose, endPose)

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

    x_star_vision_withou_blur = np.array([-5.00000000e+00, -5.00000000e+00, -5.00000000e+00, -5.00000000e+00,
       -5.00000000e+00, -5.43507098e+00, -5.86642989e+00, -5.35583304e+00,
       -3.83889073e+00, -2.12506730e+00, -1.02479773e+00, -4.74662937e-01,
        7.30131258e-19,  1.21195549e-19,  6.90565051e-21,  2.16667119e-34,
       -1.44444746e-34, -3.00000000e+01, -3.00000000e+01, -3.00000000e+01,
       -3.00000000e+01, -3.00000000e+01, -2.91261460e+01, -2.65380979e+01,
       -2.22693695e+01, -1.72273290e+01, -1.19124937e+01, -7.50736795e+00,
       -4.87385402e+00, -4.00000000e+00, -4.00000000e+00, -4.00000000e+00,
       -4.00000000e+00, -4.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.54787359e+00,
        2.31764035e+00,  2.37352904e+00,  1.65362249e+00,  9.69857499e-01,
        1.09561989e+00,  1.93044147e+00,  2.50000000e+00,  2.50000000e+00,
        2.50000000e+00,  2.50000000e+00,  2.50000000e+00,  1.04719755e+00,
        1.04719755e+00,  1.04719755e+00,  1.50508052e+00,  1.37285928e+00,
        1.67077554e+00,  1.53399944e+00,  1.05692842e+01])
    
    x_star_min_T_only = np.array([-5.00000000e+00, -5.00000000e+00, -5.00000000e+00, -5.00000000e+00,
       -5.00000000e+00, -4.35306372e+00, -3.23343811e+00, -2.55190110e+00,
       -2.50000000e+00, -2.44809890e+00, -1.76656189e+00, -6.46936279e-01,
        2.53242876e-22,  7.68207391e-23,  1.71684090e-23, -2.46519033e-32,
       -1.52536839e-30, -3.00000000e+01, -3.00000000e+01, -3.00000000e+01,
       -3.00000000e+01, -3.00000000e+01, -2.91788168e+01, -2.66470419e+01,
       -2.23364503e+01, -1.70000000e+01, -1.16635497e+01, -7.35295806e+00,
       -4.82118323e+00, -4.00000000e+00, -4.00000000e+00, -4.00000000e+00,
       -4.00000000e+00, -4.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.34327500e+00,
        1.59386875e+00,  1.65625000e+00,  1.75000000e+00,  1.84375000e+00,
        1.90613125e+00,  2.15672500e+00,  2.50000000e+00,  2.50000000e+00,
        2.50000000e+00,  2.50000000e+00,  2.50000000e+00,  1.04719755e+00,
        1.04719755e+00,  1.04719755e+00,  1.30899694e+00,  1.39626340e+00,
        1.48352986e+00,  1.57079633e+00,  1.04062887e+01]) 

    x_star20 = np.array([-5.00000000e+00, -5.00000000e+00, -5.00000000e+00, -5.00000000e+00,
       -5.00000000e+00, -5.00000000e+00, -5.06111287e+00, -4.88164002e+00,
       -4.05158841e+00, -2.47659487e+00, -6.71398742e-01,  6.64082425e-01,
        1.06867719e+00,  7.43884562e-01,  2.33561777e-01,  1.00148636e-15,
        1.48774374e-16, -6.44876407e-18, -8.16810787e-18,  2.96966179e-19,
       -1.06298781e-19, -3.00000000e+01, -3.00000000e+01, -3.00000000e+01,
       -3.00000000e+01, -3.00000000e+01, -3.00000000e+01, -2.95760757e+01,
       -2.81349094e+01, -2.53377143e+01, -2.15025316e+01, -1.71803057e+01,
       -1.27309609e+01, -8.74732570e+00, -5.88209864e+00, -4.42392432e+00,
       -4.00000000e+00, -4.00000000e+00, -4.00000000e+00, -4.00000000e+00,
       -4.00000000e+00, -4.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
        1.25513056e+00,  1.85172847e+00,  2.40817532e+00,  2.49882177e+00,
        2.07791191e+00,  1.56992940e+00,  1.44567313e+00,  1.79833291e+00,
        2.27488172e+00,  2.50000000e+00,  2.50000000e+00,  2.50000000e+00,
        2.50000000e+00,  2.50000000e+00,  2.50000000e+00,  1.04719755e+00,
        1.04719755e+00,  1.04719755e+00,  1.70621615e+00,  1.63947666e+00,
        1.55488693e+00,  1.57390162e+00,  9.41775528e+00])

    if not skip_opt:
        print('opt started...')

        num = 100
        ti_list = np.linspace(0, 1, num, endpoint=True)
        alpha1 = 0.5
        # alpha2 = 10.
        args = (P, ti_list, Rc_B, Oc_B, M[0], camera_k0, alpha1)
        res = optimize.minimize(fun=objectiveFunction, args=args, x0=x0, method='SLSQP', bounds=bounds, constraints=constraints,
                        tol=1e-6, options={'maxiter': 1e8, 'disp': True})
        print(res)
        x_star = res.x
    else:
        # x_star = x_star_took_years_to_get
        x_star = x_star20

    T = x_star[-numT:].sum()
    update_rate = 1000
    num = int(T * update_rate)
    print('num=', num)
    ti_list = np.linspace(0, T, num, endpoint=True)
    mI_loss, mC, Rw_B, Ob_W = compute_mC(x_star, P, ti_list, Rc_B, Oc_B, M[0], camera_k0)
    print(mI_loss)
    return mC, Rw_B, Ob_W


if __name__=='__main__':
    solve_n_D_OptimizationProblem()

def debug_constarints(x0, constraints):
    for idx, cons in enumerate(constraints):
        if cons['type'] == 'ineq':
            print('const', idx)
            print('ith der', cons['args'][1])
            print(cons['fun'](x0, *cons['args']))
            print()
    print(x0)