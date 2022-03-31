from casadi import *



# Piecewise-linear
x = MX(DM([1,3,7,8]))

# Between 1 and 3: 0+v+v**2
# Between 3 and 7: 3-v-v**2
# Between 7 and 8: 4*v**2
y = MX(DM([[0,3,0],[1,-1,0],[1,-1,4]]))
print(y)

n = y.shape[0]
print(n)

v = MX.sym("v")

L = low(x,v)

print(L)


coeff = y[:,L]

res = dot(coeff,v**DM(range(n)))

f = Function('f',[v],[res])


for ve in [0,1,2,3,4,7,7.5,8,8.5]:
  print(ve, f(ve), [0+ve+ve**2,3-ve-ve**2,4*ve**2] )