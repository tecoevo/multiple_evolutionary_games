import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

import egtsimplex


def f1(y,t):
    
    A = np.array([[-9.30,3.83,3.86,-1.03,-1.00,-0.96,0.10,0.33,0.16,0.20], [0.10,-1.03,0.13,3.83,-1.00,0.16,-9.30,4.06,-0.96,0.20], [0,0,0,0,0,0,0,0.20,0,0] ],dtype = float) # 4 player

    f1 = ((y[0]**3) * A[0][0]) + (3* y[0]* y[0] * y[1] * A[0][1]) + (3* y[0]* y[0] * y[2] * A[0][2])+ (3* y[0]* y[1] * y[1] * A[0][3])+ (6* y[0]* y[1] * y[2] * A[0][4])+ (3* y[0]* y[2] * y[2] * A[0][5])+ ( (y[1]**3) * A[0][6])+ (3* y[1]* y[1] * y[2] * A[0][7])+ (3* y[1]* y[2] * y[2] * A[0][8])+ ( (y[2]**3) * A[0][9])
    f2 = ((y[0]**3) * A[1][0]) + (3* y[0]* y[0] * y[1] * A[1][1]) + (3* y[0]* y[0] * y[2] * A[1][2])+ (3* y[0]* y[1] * y[1] * A[1][3])+ (6* y[0]* y[1] * y[2] * A[1][4])+ (3* y[0]* y[2] * y[2] * A[1][5])+ ( (y[1]**3) * A[1][6])+ (3* y[1]* y[1] * y[2] * A[1][7])+ (3* y[1]* y[2] * y[2] * A[1][8])+ ( (y[2]**3) * A[1][9])
    f3 = ((y[0]**3) * A[2][0]) + (3* y[0]* y[0] * y[1] * A[2][1]) + (3* y[0]* y[0] * y[2] * A[2][2])+ (3* y[0]* y[1] * y[1] * A[2][3])+ (6* y[0]* y[1] * y[2] * A[2][4])+ (3* y[0]* y[2] * y[2] * A[2][5])+ ( (y[1]**3) * A[2][6])+ (3* y[1]* y[1] * y[2] * A[2][7])+ (3* y[1]* y[2] * y[2] * A[2][8])+ ( (y[2]**3) * A[2][9])

    phi = (y[0] * f1) + (y[1] * f2) + (y[2] *f3)

    xx1 = y[0] * ( f1-phi)
    xx2 = y[1] * ( f2-phi)
    xx3 = y[2] * ( f3-phi)

    xx = np.array([xx1])

    yy = np.array([xx2] )

    zz = 1 - ( xx+yy)

    diff =  np.concatenate([xx,yy,zz])
    return xx1,xx2,xx3
    
    

t = np.arange(0,60,0.01)

#initialize simplex_dynamics object with function
dynamics1=egtsimplex.simplex_dynamics(f1)


#plot the simplex dynamics
fig,ax=plt.subplots()
dynamics1.plot_simplex(ax)
plt.show()    

