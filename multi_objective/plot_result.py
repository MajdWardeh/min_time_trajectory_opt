import os
import numpy as np
import yaml
import matplotlib.pyplot as plt


def main():
    workdir = '/home/majd/papers/Python-B-spline-examples/multi_objective/opt_alpha2_swap'
    file_name = 'opt_alpha2(blur_loss)_swap_20220315-170412.yaml'

    with open(os.path.join(workdir, file_name), 'r') as stream:
        opt_res = yaml.safe_load(stream)
    
    alpha2_list = list(opt_res.keys())
    alpha2_list.sort()

    T_loss_list = []
    blur_loss_list = []
    for alpha2 in alpha2_list[:]:
        T_loss = opt_res[alpha2]['T_loss']
        blur_loss = opt_res[alpha2]['blur_loss']
        T_loss_list.append(T_loss)
        blur_loss_list.append(blur_loss)
        print('{} {} {}'.format(alpha2, T_loss, blur_loss))
        # print(opt_res[alpha2]['x'])


    plt.plot(T_loss_list, blur_loss_list)
    plt.show()



if __name__ == '__main__':
    main()