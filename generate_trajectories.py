import os
import yaml
import time

import numpy as np
from datetime import datetime
import multiprocessing

from scipy_opt_reduced_opt_vars_traversing import solve_n_D_OptimizationProblem
from random_poses_generator import generate_rand_poses

def solve_opt(start_pose, alpha1, alpha2, que):
    mC, Rw_B, Ob_W, mI_loss, blur_loss, T, opt_res, P = solve_n_D_OptimizationProblem(startPose=start_pose, alpha1=alpha1, alpha2=alpha2)
    if opt_res is not None:
        opt_dict = {
            'P': P.getDictionary(opt_res.x),
            'x': opt_res.x.tolist(),
            'loss': float(opt_res.fun),
            'T_loss': float(T),
            'mI_loss': float(mI_loss),
            'blur_loss': float(blur_loss),
            'success': bool(opt_res.success),
            'alpha1': float(alpha1),
            'alpha2': float(alpha2),
            'start_pose': start_pose.tolist(),
        }
        key = (tuple(start_pose.tolist()), alpha1, alpha2)
        que.put((key, opt_dict))
    return None

def generate_trajectory(poses_num=10):
    que = multiprocessing.Queue()
    threads_list = []
    alpha1 = 0.05
    alpha2 = 0.001
    start_poses_list = generate_rand_poses(poses_num)
    start_time = time.time()
    for pose in start_poses_list:
        t = multiprocessing.Process(target=solve_opt, args=(pose, alpha1, alpha2, que, ))
        threads_list.append(t)
        t.start()

    for t in threads_list:
        t.join()

    print('finished in', time.time()-start_time)
    result_dict = {}
    counter = 0
    while not que.empty():
        que_return = que.get()
        counter += 1
        key, opt_dict = que_return
        result_dict[key] = opt_dict
    print('que.size:', counter)


    workdir = '/home/majd/papers/Python-B-spline-examples/multi_objective/random_trajectories'
    file_name = 'rand_traj_poses{}_alpha1{}_alpha2{}_{}.yaml'.format(poses_num, alpha1, alpha2, datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(os.path.join(workdir, file_name), 'w') as f:
        yaml.dump(result_dict, f, default_flow_style=True)

def main():
    numOfFiles = 10
    numOfPoses = 50
    for _ in range(numOfFiles):
        generate_trajectory(numOfPoses)

if __name__ == '__main__':
    main()