import numpy as np
from scipy.interpolate import BSpline
from casadi import *


n = 12
d = 4

knots=np.linspace(0,1,n-d+2,endpoint=True)
knots=np.append([0]*d,knots)
knots=np.append(knots,[1]*d)
knots = knots.tolist()

coeff = np.linspace(-10, 10, n+1, endpoint=True).tolist()

dimention = 1

x = MX.sym('x')
c = MX.sym('c', n+1)

Px1 = bspline(x, c, [knots], [d], dimention, {})

casadi_bspl = Function('Px1', [x, c], [Px1])
scipy_bspl = BSpline(knots, coeff, d)

for ti in np.linspace(0, 1, num=10, endpoint=True):
    print(scipy_bspl(ti), casadi_bspl(ti, coeff))










