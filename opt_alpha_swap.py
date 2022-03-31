import os
import yaml
import time

import numpy as np
from datetime import datetime
import multiprocessing

from scipy_opt_reduced_opt_vars_traversing import solve_n_D_OptimizationProblem

def solve_opt(alpha, alpha2, que):
    mC, Rw_B, Ob_W, mI_loss, blur_loss, T, opt_res = solve_n_D_OptimizationProblem(alpha=alpha, alpha2=alpha2)
    if opt_res is not None:
        opt_dict = {
            'x': opt_res.x.tolist(),
            'loss': float(opt_res.fun),
            'T_loss': float(T),
            'mI_loss': float(mI_loss),
            'blur_loss': float(blur_loss),
            'success': bool(opt_res.success),
            'alpha': float(alpha)
        }
        que.put((alpha2, opt_dict))
    return None

def main():
    que = multiprocessing.Queue()
    threads_list = []
    alpha = 0.05
    alpha2_list = np.linspace(0, 0.001, num=10)
    start_time = time.time()
    for alpha2 in alpha2_list:
        t = multiprocessing.Process(target=solve_opt, args=(alpha, float(alpha2), que, ))
        threads_list.append(t)
        t.start()

    for t in threads_list:
        t.join()

    print('finished swap in', time.time()-start_time)
    result_dict = {}
    counter = 0
    while not que.empty():
        que_return = que.get()
        counter += 1
        alpha2, opt_dict = que_return
        result_dict[alpha2] = opt_dict
    print('que.size:', counter)


    workdir = '/home/majd/papers/Python-B-spline-examples/multi_objective/opt_alpha2_swap'
    file_name = 'opt_alpha2(blur_loss)_swap_{}.yaml'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(os.path.join(workdir, file_name), 'w') as f:
        yaml.dump(result_dict, f, default_flow_style=True)


if __name__ == '__main__':
    main()