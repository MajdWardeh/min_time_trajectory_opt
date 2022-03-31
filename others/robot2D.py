from urllib import robotparser
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from math import sin, cos
from scipy.spatial.transform import Rotation as R

def transform(pose, vect):
    x, y, theta = pose
    # vect1 = np.append(vect, [0]).reshape((-1, 1))
    # R1 = R.from_euler('Z', -theta).as_dcm()
    vect1 = vect.reshape((-1, 1))
    R1 = np.array([
                [cos(theta), sin(theta)],
                [-sin(theta), cos(theta)]])
    
    v = np.matmul(R1, vect1-np.array([x, y]).reshape(-1, 1))
    return v

def isInFOV(robotPose, fov, Vw):
    '''
        robotPose: has the form (x, y, theta), theta is in radians.
        fov: the field of view of the robot camera in radians.
        Vw: the coordinates of the feature in the world frame.
    '''
    Vr = transform(robotPose, Vw)
    return abs(Vr[1]) <= la.norm(Vr)*sin(fov/2)

def main():

    x, y, theta = 1, 2, np.pi/3
    fov = 90*np.pi/360.0
    mag = 9
    robotPose = (x, y, theta)
    M = np.array([(6, 4), (7, 3), (x, y)])
    


    plt.plot(x, y, 'o')
    plt.arrow(x, y, mag*cos(theta+fov/2), mag*sin(theta+fov/2))
    plt.arrow(x, y, mag*cos(theta-fov/2), mag*sin(theta-fov/2))
    for m in M:
        plt.plot(m[0], m[1], 'g*' if isInFOV(robotPose, fov, m) else 'r*')
    plt.xlim((0, 10))
    plt.ylim((0, 10))

    plt.show()


if __name__ == '__main__':
    main()