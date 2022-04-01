import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from random_poses_generator import generate_rand_poses
from scipy.spatial.transform import Rotation as rot
from tf_senderV2 import TF_Sender


def test_generate_random_poses_with_FG():
    poses_num = 1000
    update_rate = 1000.0
    rand_poses = generate_rand_poses(poses_num)
    xyz_list = rand_poses[:, 0:3]
    theta_list = rand_poses[:, -1]
    pose_list = []
    for i in range(poses_num):
        p = xyz_list[i].tolist()
        q = rot.from_euler('xyz', [0, 0, theta_list[i]]).as_quat().tolist()
        for _ in range(int(update_rate/2)):
            pose_list.append(p+q)

    tf_sender = TF_Sender(pose_list, pose_update_rate=update_rate)



if __name__ == '__main__':
    test_generate_random_poses_with_FG()