from turtle import end_poly
import numpy as np
from numpy import cos, sin


def getVelocity(x_opt, P, ti_list, m, k, g=9.806):
    '''
        get the velocity of the feature m in the image plane on the x and y axes.
        args:
            x: the opt var containing the control points vars and the time vars.
            P: Trajectory object.
            ti_list: float or np array containing the time stamps that we want to compute the velocities at.
            m: np array containing the feature(s) location in the World frame.
            k: the camera calibration matrix. Actaully it is not needed, it is for providing fx and fy, which could be set to any value.
            g: the gravitational acceleration.
        The requrired velocities are computed using equations from the differential flatness property of the quadrotor.
        The equations were found using Mathematica and ported to Python using "FortranForm".
    '''
    x, y, z, theta = P.diP_dt(x_opt, 0, ti_list, axis=-1)
    x_dt, y_dt, z_dt, theta_dt = P.diP_dt(x_opt, 1, ti_list, axis=-1)
    x_d2t, y_d2t, z_d2t, theta_d2t = P.diP_dt(x_opt, 2, ti_list, axis=-1)
    x_d3t, y_d3t, z_d3t, theta_d3t = P.diP_dt(x_opt, 3, ti_list, axis=-1)
    mx, my, mz = m
    fx, fy = k[0, 0], k[1, 1]
    vel_x = (fx*(2*((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2)*(sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)*(g*mx*cos(theta)*theta_dt + g*my*sin(theta)*theta_dt + sin(theta)*z_dt*(x_d2t) - mz*cos(theta)*theta_dt*(x_d2t) + cos(theta)*z*theta_dt*(x_d2t) - cos(theta)*z_dt*(y_d2t) - mz*sin(theta)*theta_dt*(y_d2t) + sin(theta)*z*theta_dt*(y_d2t) + mx*cos(theta)*theta_dt*(z_d2t) + my*sin(theta)*theta_dt*(z_d2t) - sin(theta)*x_dt*(g + (z_d2t)) + cos(theta)*y_dt*(g + (z_d2t)) - cos(theta)*x*theta_dt*(g + (z_d2t)) - sin(theta)*y*theta_dt*(g + (z_d2t)) - mz*sin(theta)*x_d3t + sin(theta)*z*x_d3t + mz*cos(theta)*y_d3t - cos(theta)*z*y_d3t - my*cos(theta)*z_d3t + mx*sin(theta)*z_d3t - sin(theta)*x*z_d3t + cos(theta)*y*z_d3t) + 2*(g*my*cos(theta) - g*mx*sin(theta) + mz*sin(theta)*(x_d2t) - sin(theta)*z*(x_d2t) - mz*cos(theta)*(y_d2t) + cos(theta)*z*(y_d2t) + my*cos(theta)*(z_d2t) - mx*sin(theta)*(z_d2t) + sin(theta)*x*(g + (z_d2t)) - cos(theta)*y*(g + (z_d2t)))*(sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)*((x_d2t)*x_d3t + (y_d2t)*y_d3t + (g + (z_d2t))*z_d3t) + (g*my*cos(theta) - g*mx*sin(theta) + mz*sin(theta)*(x_d2t) - sin(theta)*z*(x_d2t) - mz*cos(theta)*(y_d2t) + cos(theta)*z*(y_d2t) + my*cos(theta)*(z_d2t) - mx*sin(theta)*(z_d2t) + sin(theta)*x*(g + (z_d2t)) - cos(theta)*y*(g + (z_d2t)))*(((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2)*(theta_dt*(sin(2*theta)*(x_d2t)**2 - 2*cos(2*theta)*(x_d2t)*(y_d2t) - sin(2*theta)*(y_d2t)**2) - sin(2*theta)*(y_d2t)*x_d3t + 2*cos(theta)**2*(y_d2t)*y_d3t + 2*sin(theta)*(x_d2t)*(sin(theta)*x_d3t - cos(theta)*y_d3t) + 2*g*z_d3t + 2*(z_d2t)*z_d3t) - 2*(sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)*((x_d2t)*x_d3t + (y_d2t)*y_d3t + (g + (z_d2t))*z_d3t))))/(2.*((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2)**2.5*((sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)/((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2))**1.5)
    vel_y = (fy*(2*(x_d2t**2 + y_d2t**2 + (g + z_d2t)**2)*(x_dt*x_d2t + y_dt*y_d2t + z_dt*(g + z_d2t) + (-mx + x)*x_d3t - my*y_d3t + y*y_d3t - mz*z_d3t + z*z_d3t) - 2*(-(g*mz) + (-mx + x)*x_d2t - my*y_d2t + y*y_d2t - mz*z_d2t + z*(g + z_d2t))*(x_d2t*x_d3t + y_d2t*y_d3t + (g + z_d2t)*z_d3t)))/(2.*(x_d2t**2 + y_d2t**2 + (g + z_d2t)**2)**1.5) 
    return vel_x, vel_y

def getVelocity_testing(x, y, z, theta, x_dt, y_dt, z_dt, x_d2t, y_d2t, z_d2t, theta_dt, x_d3t, y_d3t, z_d3t, g, mx, my, mz, fx, fy):
    vel_x = (fx*(2*((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2)*(sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)*(g*mx*cos(theta)*theta_dt + g*my*sin(theta)*theta_dt + sin(theta)*z_dt*(x_d2t) - mz*cos(theta)*theta_dt*(x_d2t) + cos(theta)*z*theta_dt*(x_d2t) - cos(theta)*z_dt*(y_d2t) - mz*sin(theta)*theta_dt*(y_d2t) + sin(theta)*z*theta_dt*(y_d2t) + mx*cos(theta)*theta_dt*(z_d2t) + my*sin(theta)*theta_dt*(z_d2t) - sin(theta)*x_dt*(g + (z_d2t)) + cos(theta)*y_dt*(g + (z_d2t)) - cos(theta)*x*theta_dt*(g + (z_d2t)) - sin(theta)*y*theta_dt*(g + (z_d2t)) - mz*sin(theta)*x_d3t + sin(theta)*z*x_d3t + mz*cos(theta)*y_d3t - cos(theta)*z*y_d3t - my*cos(theta)*z_d3t + mx*sin(theta)*z_d3t - sin(theta)*x*z_d3t + cos(theta)*y*z_d3t) + 2*(g*my*cos(theta) - g*mx*sin(theta) + mz*sin(theta)*(x_d2t) - sin(theta)*z*(x_d2t) - mz*cos(theta)*(y_d2t) + cos(theta)*z*(y_d2t) + my*cos(theta)*(z_d2t) - mx*sin(theta)*(z_d2t) + sin(theta)*x*(g + (z_d2t)) - cos(theta)*y*(g + (z_d2t)))*(sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)*((x_d2t)*x_d3t + (y_d2t)*y_d3t + (g + (z_d2t))*z_d3t) + (g*my*cos(theta) - g*mx*sin(theta) + mz*sin(theta)*(x_d2t) - sin(theta)*z*(x_d2t) - mz*cos(theta)*(y_d2t) + cos(theta)*z*(y_d2t) + my*cos(theta)*(z_d2t) - mx*sin(theta)*(z_d2t) + sin(theta)*x*(g + (z_d2t)) - cos(theta)*y*(g + (z_d2t)))*(((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2)*(theta_dt*(sin(2*theta)*(x_d2t)**2 - 2*cos(2*theta)*(x_d2t)*(y_d2t) - sin(2*theta)*(y_d2t)**2) - sin(2*theta)*(y_d2t)*x_d3t + 2*cos(theta)**2*(y_d2t)*y_d3t + 2*sin(theta)*(x_d2t)*(sin(theta)*x_d3t - cos(theta)*y_d3t) + 2*g*z_d3t + 2*(z_d2t)*z_d3t) - 2*(sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)*((x_d2t)*x_d3t + (y_d2t)*y_d3t + (g + (z_d2t))*z_d3t))))/(2.*((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2)**2.5*((sin(theta)**2*(x_d2t)**2 - sin(2*theta)*(x_d2t)*(y_d2t) + cos(theta)**2*(y_d2t)**2 + (g + (z_d2t))**2)/((x_d2t)**2 + (y_d2t)**2 + (g + (z_d2t))**2))**1.5)
    vel_y = (fy*(2*(x_d2t**2 + y_d2t**2 + (g + z_d2t)**2)*(x_dt*x_d2t + y_dt*y_d2t + z_dt*(g + z_d2t) + (-mx + x)*x_d3t - my*y_d3t + y*y_d3t - mz*z_d3t + z*z_d3t) - 2*(-(g*mz) + (-mx + x)*x_d2t - my*y_d2t + y*y_d2t - mz*z_d2t + z*(g + z_d2t))*(x_d2t*x_d3t + y_d2t*y_d3t + (g + z_d2t)*z_d3t)))/(2.*(x_d2t**2 + y_d2t**2 + (g + z_d2t)**2)**1.5) 
    return vel_x, vel_y

def x_(t):
    return t**4 + t**3 + t**2 + t + 1

def x_dt_(t):
    return 4*t**3 + 3*t**2 + 2*t + 1 

def x_d2t_(t):
    return 12*t**2 + 6*t + 2 

def x_d3t_(t):
    return 24*t + 6 

def x_d4t_(t):
    return 24. 

def theta_(t):
    return t**2 + t + 5

def theta_dt_(t):
    return 2.*t + 1

def testing():
    t = 2

    x = x_(t)
    x_dt = x_dt_(t)
    x_d2t = x_d2t_(t)
    x_d3t = x_d3t_(t)

    y = x_(t)
    y_dt = x_dt_(t)
    y_d2t = x_d2t_(t)
    y_d3t = x_d3t_(t)

    z = x_(t)
    z_dt = x_dt_(t)
    z_d2t = x_d2t_(t)
    z_d3t = x_d3t_(t)

    theta = theta_(t)
    theta_dt = theta_dt_(t)


    vel_ = getVelocity_testing(x, y, z, theta, x_dt, y_dt, z_dt, x_d2t, y_d2t, z_d2t, theta_dt, x_d3t, y_d3t, z_d3t, mx=0, my=0, mz=2.9, g=9.806, fx=10., fy=10.)
    # vel = getVelocity(0, y, 0, theta, 0, y_dt, 0, 0, y_d2t, 0, theta_dt, 0, y_d3t, 0)
    # vel = getVelocity(x, 0, 0, theta, x_dt, 0, 0, x_d2t, 0, 0, theta_dt, x_d3t, 0, 0)
    # vel = getVelocity(0, 0, z, theta, 0, 0, z_dt, 0, 0, z_d2t, theta_dt, 0, 0, z_d3t)
    # vel = getVelocity(x, 0, z, theta, x_dt, 0, z_dt, x_d2t, 0, z_d2t, theta_dt, x_d3t, 0, z_d3t)
    # vel = getVelocity(0, y, z, theta, 0, y_dt, z_dt, 0, y_d2t, z_d2t, theta_dt, 0, y_d3t, z_d3t)
    # vel = getVelocity(x, y, 0.0, theta, x_dt, y_dt, 0.0, x_d2t, y_d2t, 0.0, theta_dt, x_d3t, y_d3t, 0.0)
    print(vel_)




if __name__ == '__main__':
    testing()