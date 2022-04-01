import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

def generate_rand_poses(poses_num, debugging=False):
    gate_center = np.array([0.0, 0.0, 2.03849])
    gate_corners = np.array([[0.9014978, 0.001497865, 2.961498], [-0.898504, 0.001497627, 2.961498], [0.9014994, 0.001497865, 1.115498], [-0.898504, 0.001497627, 1.115498]])
    P1, P2 = gate_corners[[0, 2], :].mean(axis=0), gate_corners[[1, 3], :].mean(axis=0)

    horizontal_FOV = np.deg2rad(90.0)
    rotation_discount = 0.95
    
    xyz_min_max = np.array([[-5, 5], [-35, -2], [1, 3]])
    initial_heading_max_abs = np.deg2rad(45)

    poses_dict = {}
    xyz_list = []
    theta_list = []
    for i in range(poses_num):
        ## generating random position xyz
        ## initial_heading is zero, when the drone is facing the gate with zeros angle
        initial_heading = np.pi/2
        while np.abs(initial_heading) >  initial_heading_max_abs:
            xyz = np.random.rand(3) * (xyz_min_max[:, 1] - xyz_min_max[:, 0]) + xyz_min_max[:, 0]
            initial_heading = np.arctan2(gate_center[1]-xyz[1], gate_center[0]-xyz[0]) - np.pi/2

        ## computing heading

        ## alphas are the FOV/2 angles
        alpha1 = initial_heading - horizontal_FOV/2
        alpha2 = initial_heading + horizontal_FOV/2

        P1_angle = np.arctan2(P1[1]-xyz[1], P1[0]-xyz[0]) - np.pi/2
        P2_angle = np.arctan2(P2[1]-xyz[1], P2[0]-xyz[0]) - np.pi/2

        rot1 = P1_angle - alpha1 # must be always postive
        rot2 = P2_angle - alpha2 # must be always negative

        assert rot1 > 0 and rot2 < 0
        max_rot_angle = rotation_discount * (rot1 if rot1 <= abs(rot2) else rot2)
        random_rot_angle = np.random.rand() * (max_rot_angle * 2) - max_rot_angle 

        theta = initial_heading + random_rot_angle

        ## quick fix: add pi/2 to theta
        theta += np.pi/2
        
        ## adding to the list
        xyz_list.append(xyz)
        theta_list.append(theta)

        ## debugging
        if debugging:
            plot(xyz, gate_center, P1, P2, initial_heading, alpha1, alpha2, P1_angle, P2_angle, random_rot_angle, theta, xyz_min_max)

    xyz_list = np.array(xyz_list)
    theta_list = np.array(theta_list).reshape(-1, 1)
    return np.concatenate([xyz_list, theta_list], axis=1)

def plot(xyz, gate_center, P1, P2, initial_heading, alpha1, alpha2, P1_angle, P2_angle, random_rot_angle, theta, xyz_min_max):
    plt.plot(gate_center[0], gate_center[1], 'bo')
    plt.plot(P1[0], P1[1], 'bo')
    plt.plot(P2[0], P2[1], 'bo')
    plt.plot(xyz[0], xyz[1], 'ro')

    plt.arrow(xyz[0], xyz[1], gate_center[0]-xyz[0], gate_center[1]-xyz[1], color='b')
    plt.arrow(xyz[0], xyz[1], 100*np.cos(P1_angle+np.pi/2), 100*np.sin(P1_angle+np.pi/2), color='r')
    plt.arrow(xyz[0], xyz[1], 100*np.cos(P2_angle+np.pi/2), 100*np.sin(P2_angle+np.pi/2), color='r')
    ## alphas
    plt.arrow(xyz[0], xyz[1], np.cos(alpha1+np.pi/2), np.sin(alpha1+np.pi/2), color='g')
    plt.arrow(xyz[0], xyz[1], np.cos(alpha2+np.pi/2), np.sin(alpha2+np.pi/2), color='g')

    ## rotated angles
    plt.arrow(xyz[0], xyz[1], np.cos(theta+np.pi/2), np.sin(theta+np.pi/2), color='y')
    plt.arrow(xyz[0], xyz[1], 100*np.cos(alpha1+random_rot_angle+np.pi/2), 100*np.sin(alpha1+random_rot_angle+np.pi/2), color='y')
    plt.arrow(xyz[0], xyz[1], 100*np.cos(alpha2+random_rot_angle+np.pi/2), 100*np.sin(alpha2+random_rot_angle+np.pi/2), color='y')

    plt.xlim(xyz_min_max[0, 0], xyz_min_max[0, 1])
    plt.ylim(xyz_min_max[1, 0], 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def test_generate_random_poses():
    rand_poses = generate_rand_poses(10000)
    xyz_list = rand_poses[:, 0:3]
    theta_list = rand_poses[:, -1]
    theta_list_deg = np.rad2deg(theta_list)
    print('xyz: avg={}, min={}, max={}'.format(xyz_list.mean(axis=0), xyz_list.min(axis=0), xyz_list.max(axis=0)))
    print('theta (deg): avg={}, min={}, max={}'.format(theta_list_deg.mean(), theta_list_deg.min(), theta_list_deg.max()))

def debugging():
    rand_poses = generate_rand_poses(1)
    xyz = rand_poses[:, 0:3]
    theta_deg = np.rad2deg(rand_poses[:, -1])
    print(xyz, theta_deg)



if __name__ == '__main__':
    test_generate_random_poses()