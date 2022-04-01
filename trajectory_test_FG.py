import os
import time
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as rot
from scipy_opt_reduced_opt_vars_traversing import solve_n_D_OptimizationProblem, compute_mC, Trajectory

from tf_senderV2 import TF_Sender

def get_x_star_from_file():
    workdir = '/home/majd/papers/Python-B-spline-examples/multi_objective/opt_alpha_swap'
    file_name = 'opt_alpha2(blur_loss)_swap_20220315-170412.yaml'

    with open(os.path.join(workdir, file_name), 'r') as stream:
        opt_res = yaml.safe_load(stream)

    alpha_list = list(opt_res.keys())
    alpha_list.sort()

    alpha = alpha_list[1]
    x_star = np.array(opt_res[alpha]['x'])
    return alpha, x_star

def get_x_star():
    # x_star = np.array([ -4.94938041,  -2.9414649 ,  -0.04005703,   1.39339381,
    #      1.28817034,   0.19870859,  -0.18431368, -27.39379261,
    #    -23.02114744, -17.9503781 , -11.34327771,  -4.89532009,
    #      0.53751953,   4.36063421,   1.87572244,   3.02520432,
    #      2.46816085,   1.6196792 ,   1.15383925,   1.30919439,
    #      2.50212491,   1.60672248,   1.42075396,   1.6858234 ,
    #      1.70364992,   1.6616723 ,  10.51660569,   0.80044628])
    x_star = np.array([-4.11603374e+00, -2.56821265e+00, -1.60279428e+00, -1.36195851e+00,
       -8.83807536e-01, -2.07222336e-02,  2.70838861e-01, -2.88959223e+01,
       -2.53898058e+01, -1.92877670e+01, -1.15000000e+01, -3.71223303e+00,
        2.38980581e+00,  5.89592232e+00,  9.73933829e-01,  1.49289561e+00,
        2.02390181e+00,  1.91642026e+00,  1.62399652e+00,  1.91088733e+00,
        2.43727174e+00,  1.61195110e+00,  1.40962929e+00,  1.52777603e+00,
        1.42338466e+00,  1.49742229e+00,  8.51442740e+00])
    alpha = 0.05
    return alpha, x_star

def test_opt():
    alpha, x_star = get_x_star()
    mC, Rw_B, Ob_W, mI_loss, blur_loss, T, opt_res, P = solve_n_D_OptimizationProblem(x_star=x_star)
    print('T_loss:', T)
    print('alpha:', alpha)
    print('mI_loss:', mI_loss)
    print('blur_loss:', blur_loss)
    pose_list = []
    for i in range(mC.shape[0]):
        Rb_W = Rw_B[i].T
        p = Ob_W[i].squeeze().tolist()
        q = rot.from_dcm(Rb_W).as_quat().tolist()
        pose_list.append(p+q)
    
    tf_sender = TF_Sender(pose_list, mC, pose_update_rate=1000.0)
    tf_sender = None


def test_opt_trajs_dict():
    workdir = '/home/majd/papers/Python-B-spline-examples/multi_objective/random_trajectories'
    file_name = 'rand_traj_poses1_alpha10.05_alpha20.001_20220331-230034.yaml'

    camera_k0 = np.diag([10, 10, 1])
    Rc_B = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    Oc_B = np.zeros((3, 1))
    M = np.array([[0.0, 0.0, 2.03849]]) 
    update_rate = 1000.0

    with open(os.path.join(workdir, file_name), 'r') as stream:
        trajs_dict = yaml.load(stream)
    
    traj_keys_list = list(trajs_dict.keys())

    for key in traj_keys_list[:3]:
        if trajs_dict[key]['success']:
            T_loss = trajs_dict[key]['T_loss']
            mI_loss = trajs_dict[key]['mI_loss']
            blur_loss = trajs_dict[key]['blur_loss']

            P = Trajectory.from_dictionary(trajs_dict[key]['P'])
            x_star = np.array(trajs_dict[key]['x'])

            num = int(T_loss * update_rate)
            ti_list = np.linspace(0, T_loss, num, endpoint=True)
            mI, mC, Rw_B, Ob_W = compute_mC(x_star, P, ti_list, Rc_B, Oc_B, M[0], camera_k0)

            print('{}:\n{} {} {}'.format(key, T_loss, blur_loss, mI_loss))
            print('-------------------------------')

            pose_list = []
            for i in range(mC.shape[0]):
                Rb_W = Rw_B[i].T
                p = Ob_W[i].squeeze().tolist()
                q = rot.from_dcm(Rb_W).as_quat().tolist()
                pose_list.append(p+q)
            tf_sender = TF_Sender(pose_list, mC, pose_update_rate=update_rate)
            tf_sender.kill()
            time.sleep(1)
        else:
            print('trajectory opt was not successful')
            print(key)
            print('-------------------------------')


if __name__ == '__main__':
    test_opt_trajs_dict()