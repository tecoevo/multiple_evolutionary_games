import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import time
import scipy.integrate as integrate
import scipy as sp
from scipy.integrate import odeint
import scipy.spatial
from scipy.spatial import Delaunay
from matplotlib.pyplot import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Bbox
import scipy.interpolate
import math as mt



### USE THE TERMINAL TO EXECUTE THIS # 

### FIRST RUN egtsimplex.py BEFORE RUNNING THIS CODE 

def comb1(N,k1,k2,k3):

    C = ( mt.factorial(N) / ( mt.factorial(k1) * mt.factorial(k2) * mt.factorial(k3) ) )

    return C

def comb2(N,k1,k2):

    C = ( mt.factorial(N) / ( mt.factorial(k1) * mt.factorial(k2)  ) )

    return C


def fitness(P,a1,a2):

    F = np.zeros((2,3),dtype = float)

    # number of players

    d1 = 2
    D1 = d1 - 1

    d2 = 4
    D2 = d2 - 1

    for alpha1 in range (0,d1):
        for alpha2 in range (0,d1):
            for alpha3 in range (0,d1):
                if (alpha1+alpha2+alpha3== D1):
                    binom = comb1(D1,alpha1,alpha2,alpha3)
                    P_alpha = np.power(P[0][0],alpha1) * np.power(P[0][1],alpha2) * np.power(P[0][2],alpha3)
                    if (alpha2==0 and alpha3==0):
                        index = 0
                    elif (alpha2==1 and alpha3==0):
                          index = 1
                    elif (alpha2==0 and alpha3==1):
                          index = 2
                    elif (alpha2==2 and alpha3==0):
                          index = 3
                    elif (alpha2==1 and alpha3==1):
                          index = 4
                    elif (alpha2==0 and alpha3==2):
                          index = 5
                    elif (alpha2==3 and alpha3==0):
                          index = 6
                    elif (alpha2==2 and alpha3==1):
                          index = 7
                    elif (alpha2==1 and alpha3==2):
                          index = 8
                    elif (alpha2==0 and alpha3==3):
                          index = 9
                    F[0][0] = F[0][0] + ( binom * P_alpha * a1[0][index])
                    F[0][1] = F[0][1] + ( binom * P_alpha * a1[1][index])
                    F[0][2] = F[0][2] + ( binom * P_alpha * a1[2][index])



    for alpha1 in range (0,d2):
        for alpha2 in range (0,d2):
            for alpha3 in range (0,d2):
                if (alpha1+alpha2+alpha3== D2):
                    binom = comb1(D1,alpha1,alpha2,alpha3)
                    P_alpha = np.power(P[1][0],alpha1) * np.power(P[1][1],alpha2) * np.power(P[1][2],alpha3)
                    if (alpha2==0 and alpha3==0):
                        index = 0
                    elif (alpha2==1 and alpha3==0):
                          index = 1
                    elif (alpha2==0 and alpha3==1):
                          index = 2
                    elif (alpha2==2 and alpha3==0):
                          index = 3
                    elif (alpha2==1 and alpha3==1):
                          index = 4
                    elif (alpha2==0 and alpha3==2):
                          index = 5
                    elif (alpha2==3 and alpha3==0):
                          index = 6
                    elif (alpha2==2 and alpha3==1):
                          index = 7
                    elif (alpha2==1 and alpha3==2):
                          index = 8
                    elif (alpha2==0 and alpha3==3):
                          index = 9
                    F[1][0] = F[1][0] + ( binom * P_alpha * a2[0][index])
                    F[1][1] = F[1][1] + ( binom * P_alpha * a2[1][index])
                    F[1][2] = F[1][2] + ( binom * P_alpha * a2[2][index])

    #print ('F is  ', F)
    return F


def PHI(FF,PP):

    phix = np.zeros((2,3),dtype = float)

    phix[0][0] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1]) + (PP[0][2]*FF[0][2])
    phix[0][1] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1]) + (PP[0][2]*FF[0][2])
    phix[0][2] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1]) + (PP[0][2]*FF[0][2])
    phix[1][0] = (PP[1][0]*FF[1][0]) + (PP[1][1]*FF[1][1]) + (PP[1][2]*FF[1][2])   # FF[1][2] = 0 as Game 2 has no strategy of type 3. Therefore, the third term just goes to zero
    phix[1][1] = (PP[1][0]*FF[1][0]) + (PP[1][1]*FF[1][1]) + (PP[1][2]*FF[1][2])
    phix[1][2] = (PP[1][0]*FF[1][0]) + (PP[1][1]*FF[1][1]) + (PP[1][2]*FF[1][2])

    #print ('phix is', phix)
    return  phix


def func(y,t,A1,A2):

    #print(' y is ', y)

    xx = np.array([[y[0],y[1],y[2]], [y[3],y[4],y[5]], [y[6],y[7],y[8]]],dtype = float)

    pp = np.zeros((2,3),dtype = float)

    yy = np.zeros((3,3),dtype = float)


    pp[0][0] = xx[0][0] + xx[0][1] + xx[0][2]
    pp[0][1] = xx[1][0] + xx[1][1] + xx[1][2]
    pp[0][2] = xx[2][0] + xx[2][1] + xx[2][2]
    pp[1][0] = xx[0][0] + xx[1][0] + xx[2][0]
    pp[1][1] = xx[0][1] + xx[1][1] + xx[2][1]
    pp[1][2] = xx[0][2] + xx[1][2] + xx[2][2]

    #print ('p is', pp)

    f = fitness(pp,A1,A2) # 2D array with four elements

    phi = PHI(f,pp)   # 2D array with four elements. Ensure rows have same value as for f11 and f12 --> same pi


    yy[0][0] = xx[0][0] * ( (f[0][0] - phi[0][0])  + (f[1][0]-phi[1][0]) )
    yy[0][1] = xx[0][1] * ( (f[0][0] - phi[0][0])  + (f[1][1]-phi[1][1]) )
    yy[0][2] = xx[0][2] * ( (f[0][0] - phi[0][0])  + (f[1][2]-phi[1][2]) )

    yy[1][0] = xx[1][0] * ( (f[0][1] - phi[0][1])  + (f[1][0]-phi[1][0]) )
    yy[1][1] = xx[1][1] * ( (f[0][1] - phi[0][1])  + (f[1][1]-phi[1][1]) )
    yy[1][2] = xx[1][2] * ( (f[0][1] - phi[0][1])  + (f[1][2]-phi[1][2]) )


    yy[2][0] = xx[2][0] * ( (f[0][2] - phi[0][2])  + (f[1][0]-phi[1][0]) )
    yy[2][1] = xx[2][1] * ( (f[0][2] - phi[0][2])  + (f[1][1]-phi[1][1]) )
    yy[2][2] = xx[2][2] * ( (f[0][2] - phi[0][2])  + (f[1][2]-phi[1][2]) )

    X11 = yy[0][0]
    X12 = yy[0][1]
    X13 = yy[0][2]
    X21 = yy[1][0]
    X22 = yy[1][1]
    X23 = yy[1][2]
    X31 = yy[2][0]
    X32 = yy[2][1]
    X33 = yy[2][2]

    return X11,X12,X13, X21,X22,X23, X31,X32,X33

def coordinates(a,b,c):

    Xnew = 1 - (a/2) - b
    Ynew = np.sqrt(3)/2 * a

    XN = np.array([Xnew],dtype = float)
    YN = np.array([Ynew],dtype = float)

    ret = np.concatenate([XN,YN])

    return ret

####### main #####

A_1 = np.array([[-1,10,-10],[-6,-1,6],[2,-2,-1]])
A_2 = np.array([[-9,3,3,-1,1,2],[0,0,0,0,0,0],[0,0,0,0,2,0]]) # 3 player
A_2 = np.array([[-9.30,3.83,3.86,-1.03,-1.00,-0.96,0.10,0.33,0.16,0.20], [0.10,-1.03,0.13,3.83,-1.00,0.16,-9.30,4.06,-0.96,0.20], [0,0,0,0,0,0,0,0.20,0,0] ],dtype = float) # 4 player

t = np.arange(0,100,0.01 )  # time

l = len(t)



####### initial values for x11, x12, x21, x22, x31 and x32


initial1 = np.array([ 0.01 ,0.19 ,0.1 , 0.2, 0.1, 0.1,0.1,0.1,0.1 ],dtype = float)

x110  = 0.1760
x120 = 0.065727
x130 = 0.0380991
x210 = 0.00225077
x220 = 0.076174
x230 = 0.2018188
x310 = 0.424781
x320 =  0.110879
x330 = 1 - (x110+x120 +x130 + x210+x220+x230+x310+x320)

x110  = 0.01
x120 = 0.165727
x130 = 0.0380991
x210 = 0.0022507745
x220 = 0.176174
x230 = 0.1018188
x310 = 0.324781
x320 =  0.110879
x330 = 1 - (x110+x120 +x130 + x210+x220+x230+x310+x320)

initial1 = np.array([ x110 , x120, x130, x210, x220, x230, x310, x320, x330 ],dtype = float)

initial2 = np.array([ 0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 ],dtype = float)

initial3 = np.array([ 0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.1,0.1],dtype = float)

x110  = 0.176083
x120 = 0.065727
x130 = 0.0240991
x210 = 0.00225077
x220 = 0.176174
x230 = 0.0018188
x310 = 0.224781
x320 =  0.110879

x330 = 1 - (x110+x120 +x130 + x210+x220+x230+x310+x320)

initial3 = np.array([ x110 , x120, x130, x210, x220, x230, x310, x320, x330 ],dtype = float)

sol1  = odeint(func,initial1, t, args= (A_1,A_2))

sol2  = odeint(func,initial2, t, args= (A_1,A_2))

sol3  = odeint(func,initial3, t, args= (A_1,A_2))


x11_1 = sol1[:,0]
x12_1 = sol1[:,1]
x13_1 = sol1[:,2]
x21_1 = sol1[:,3]
x22_1 = sol1[:,4]
x23_1 = sol1[:,5]
x31_1 = sol1[:,6]
x32_1 = sol1[:,7]
x33_1 = sol1[:,8]

p11_1 = x11_1 + x12_1 + x13_1
p12_1 = x21_1 + x22_1 + x23_1
p13_1 = x31_1 + x32_1 + x33_1
p21_1 = x11_1 + x21_1 + x31_1
p22_1 = x12_1 + x22_1 + x32_1
p23_1 = x13_1 + x23_1 + x33_1



x11_2 = sol2[:,0]
x12_2 = sol2[:,1]
x13_2 = sol2[:,2]
x21_2 = sol2[:,3]
x22_2 = sol2[:,4]
x23_2 = sol2[:,5]
x31_2 = sol2[:,6]
x32_2 = sol2[:,7]
x33_2 = sol2[:,8]

p11_2 = x11_2 + x12_2 + x13_2
p12_2 = x21_2 + x22_2 + x23_2
p13_2 = x31_2 + x32_2 + x33_2
p21_2 = x11_2 + x21_2 + x31_2
p22_2 = x12_2 + x22_2 + x32_2
p23_2= x13_2 + x23_2 + x33_2



x11_3 = sol3[:,0]
x12_3 = sol3[:,1]
x13_3 = sol3[:,2]
x21_3 = sol3[:,3]
x22_3 = sol3[:,4]
x23_3 = sol3[:,5]
x31_3 = sol3[:,6]
x32_3 = sol3[:,7]
x33_3 = sol3[:,8]

p11_3 = x11_3 + x12_3 + x13_3
p12_3 = x21_3 + x22_3 + x23_3
p13_3 = x31_3 + x32_3 + x33_3
p21_3 = x11_3 + x21_3 + x31_3
p22_3 = x12_3 + x22_3 + x32_3
p23_3 = x13_3 + x23_3 + x33_3




######
simplex1 = coordinates(p11_1,p12_1,p13_1)
simplex2 = coordinates(p11_2,p12_2,p13_2)
simplex3 = coordinates(p11_3,p12_3,p13_3)

equi = coordinates(0.3333333,0.3333333,0.3333333)
fig =plt.figure()
plt.rcParams.update({'font.size': 30, 'text.color':'black'})

plt.plot(simplex1[0,:],simplex1[1,:],c='blue',ms = 30,label='initial condition 1' )
plt.plot(simplex2[0,:],simplex2[1,:],c='green',ms = 30,label='initial condition 2')
plt.plot(simplex3[0,:],simplex3[1,:],c='magenta',ms = 30,label='initial condition 3')

plt.plot(simplex1[0][0],simplex1[1][0],'.', ms=3, c='orange')
plt.plot(simplex2[0][0],simplex2[1][0],'.', ms=3, c='orange')
plt.plot(simplex3[0][0],simplex3[1][0],'.', ms=3, c='orange')

plt.plot(simplex1[0][l-1],simplex1[1][l-1],'^', ms=3, c='k')
plt.plot(simplex2[0][l-1],simplex2[1][l-1],'^', ms=3, c='k')
plt.plot(simplex3[0][l-1],simplex3[1][l-1],'^', ms=3, c='k')
#plt.legend(loc =4, fontsize=10, labelspacing=0.01)
plt.plot(equi[0],equi[1],'o',ms=3,c='k')

xax1 = np.array([0.5,0])
yax1 = np.array([ 0.8660254037844386,0  ])

xax2 = np.array([0.5,1])
yax2 = np.array([ 0.8660254037844386,0  ])

xax3 = np.array([1,0.0])
yax3 = np.array([ 0.0,0.0  ])

plt.plot(xax1,yax1,'-',c='black')
plt.plot(xax2,yax2,'-',c='black')
plt.plot(xax3,yax3,'-',c='black')

lab1=[0,1,0.5]
lab2=[0,0,0.8660254037844386]

lab3=['$p_{12}$','$p_{13}$','$p_{11}$']


ax = plt.gca()
ax.scatter(lab1, lab2)

for i, txt in enumerate(lab3):
    ax.annotate(txt, (lab1[i],lab2[i]) )

ax1 = plt.gca()
ax1.axes.get_xaxis().set_ticks([])
ax1.axes.get_yaxis().set_ticks([])
#frame.patch.set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#frame.axes.get_xaxis().set_visible(False)
plt.xlim([0.0,1.0])
plt.axes().set_aspect('equal', 'datalim')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#plt.axis('off')
plt.title("MGD, simplex a ")
#plt.savefig('../vandana/multiple/Omnigraffle_figures/3x3+3x3x3x3_a.svg',dpi=300)
plt.show()



##########


simplex4 = coordinates(p21_1,p22_1,p23_1)
simplex5 = coordinates(p21_2,p22_2,p23_2)
simplex6 = coordinates(p21_3,p22_3,p23_3)


equi1 = coordinates(0.094587,0.094587,0.810826)
equi2 = coordinates(0.46685,0.46685,0.0662998)
equi3 = coordinates(0.171264,0.0843903,0.744346)
equi4 = coordinates(0.0843903,0.171264,  0.744346)
equi5 = coordinates(0.18919,0.18919,0.621621)
equi6 = coordinates(0.223325,0.476114,0.300561 )
equi7 = coordinates(0.476114,0.223325, 0.300561)
equi8 = coordinates(0.482313,0.0591478, 0.458539)
equi9 = coordinates(0.0591478,0.482313,0.458539)


fig =plt.figure()
plt.rcParams.update({'font.size': 30, 'text.color':'black'})

plt.plot(simplex4[0,:],simplex4[1,:],c='blue',ms = 30,label='initial condition 1' )
plt.plot(simplex5[0,:],simplex5[1,:],c='green',ms = 30,label='initial condition 2')
plt.plot(simplex6[0,:],simplex6[1,:],c='magenta',ms =30,label='initial condition 3')

plt.plot(simplex4[0][0],simplex4[1][0],'.', ms=3, c='orange')
plt.plot(simplex5[0][0],simplex5[1][0],'.', ms=3, c='orange')
plt.plot(simplex6[0][0],simplex6[1][0],'.', ms=3, c='orange')

plt.plot(simplex4[0][l-1],simplex4[1][l-1],'^', ms=3, c='k')
plt.plot(simplex5[0][l-1],simplex5[1][l-1],'^', ms=3, c='k')
plt.plot(simplex6[0][l-1],simplex6[1][l-1],'^', ms=3, c='k')
#plt.legend(loc =4, fontsize=10, labelspacing=0.01)

plt.plot(equi1[0],equi1[1], 'o',ms=5, c='k')
plt.plot(equi2[0],equi2[1],'o', ms=5, c='k')
plt.plot(equi3[0],equi3[1],'o', ms=5, c='silver')
plt.plot(equi4[0],equi4[1],'o', ms=5, c='silver')
plt.plot(equi5[0],equi5[1],'ko',mfc='none')
plt.plot(equi6[0],equi6[1],'o', ms=5, c='silver')
plt.plot(equi7[0],equi7[1],'o', ms=5, c='silver')
plt.plot(equi8[0],equi8[1],'o', ms=5, c='k')
plt.plot(equi9[0],equi9[1],'o', ms=5, c='k')

xax1 = np.array([0.5,0])
yax1 = np.array([ 0.8660254037844386,0  ])

xax2 = np.array([0.5,1])
yax2 = np.array([ 0.8660254037844386,0  ])

xax3 = np.array([1,0.0])
yax3 = np.array([ 0.0,0.0  ])

plt.plot(xax1,yax1,'-',c='black')
plt.plot(xax2,yax2,'-',c='black')
plt.plot(xax3,yax3,'-',c='black')

lab1=[0,1,0.5]
lab2=[0,0,0.8660254037844386]

lab3=['$p_{22}$','$p_{23}$','$p_{21}$']


ax = plt.gca()
ax.scatter(lab1, lab2)

for i, txt in enumerate(lab3):
    ax.annotate(txt, (lab1[i],lab2[i]))

ax1 = plt.gca()
ax1.axes.get_xaxis().set_ticks([])
ax1.axes.get_yaxis().set_ticks([])
#frame.patch.set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#frame.axes.get_xaxis().set_visible(False)
plt.xlim([0.0,1.0])
plt.axes().set_aspect('equal', 'datalim')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#plt.axis('off')
plt.title("MGD, simplex b ")
#plt.savefig('../vandana/multiple/Omnigraffle_figures/3x3+3x3x3x3_b.svg',dpi=300)
plt.show()

# SGD 1

def f(y,t,A):

    f1 = (y[0] * A[0][0]) + (y[1] * A[0][1]) + (y[2] * A[0][2])
    f2 = (y[0] * A[1][0]) + (y[1] * A[1][1]) + (y[2] * A[1][2])
    f3 = (y[0] * A[2][0] )+ (y[1] * A[2][1] )+ (y[2] * A[2][2])

    phi = (y[0] * f1) + (y[1] * f2) + (y[2] *f3)

    xx1 = y[0] * ( f1-phi)
    xx2 = y[1] * ( f2-phi)
    xx3 = y[2] * ( f3-phi)

    xx = np.array([xx1])

    yy = np.array([xx2] )

    zz = 1 - ( xx+yy)

    diff =  np.concatenate([xx,yy,zz])
    return xx1,xx2,xx3


def coordinates(a,b):

    #Xnew = (1 /2 )*   ( ( c + (2*b) ) / (a+b +c )       )
    #Ynew = (( np.sqrt (3) ) /  2 ) *  (  c / (a+b+c)          )

    Xnew = 1 - (a/2) - b
    Ynew = np.sqrt(3)/2 * a


    XN = np.array([Xnew],dtype = float)
    YN = np.array([Ynew],dtype = float)

    ret = np.concatenate([XN,YN])

    return ret
#payoff matrix


A = np.array([[-1,10,-10],[-6,-1,6],[2,-2,-1]])

# Initial conditions

x10  = 0.3
x20=  0.3
x30 = 1 - (x10+x20)

initial = np.array([x10,x20,x30])

initialone = np.array([0.3,0.3,0.34])

initialone = np.array([0.1,0.1,0.8])
initialtwo = np.array([0.3,0.2,0.5])

initialthree = np.array([0.2,0.2,0.6])

t = np.arange(0,60,0.01)

l = len(t)

sol_1 = odeint(f,initialone,t,args=(A,))

x1_1 = sol_1[:,0]
x2_1 = sol_1[:,1]
x3_1 = sol_1[:,2]

sol_2 = odeint(f,initialtwo,t,args=(A,))

x1_2 = sol_2[:,0]
x2_2 = sol_2[:,1]
x3_2 = sol_2[:,2]

sol_3 = odeint(f,initialthree,t,args=(A,))

x1_3 = sol_3[:,0]
x2_3 = sol_3[:,1]
x3_3 = sol_3[:,2]


equi = np.ones(l)
equilibrium = equi * 0.333


coord1 = coordinates(0,0)
coord2 = coordinates(1,0)
coord3 = coordinates(0,1)
xax1 = np.array([0.5,0])
yax1 = np.array([ 0.8660254037844386,0  ])

xax2 = np.array([0.5,1])
yax2 = np.array([ 0.8660254037844386,0  ])

xax3 = np.array([1,0.0])
yax3 = np.array([ 0.0,0.0  ])

simplex1 = coordinates(x1_1,x2_1)
simplex2 = coordinates(x1_2,x2_2)
simplex3 = coordinates(x1_3,x2_3)

attractor = coordinates(0.33,0.33)

attractor = np.array([0.505  ,  0.28578838])

fig =plt.figure()
plt.rcParams.update({'font.size': 30, 'text.color':'black'})

#fig = plt.figure(frameon=False)
#ax = fig.add_axes([1, 1, 1,1])
#ax.axis('off')
plt.plot(simplex1[0,:],simplex1[1,:],label='initial conditon 1')
plt.plot(simplex2[0,:],simplex2[1,:],label='initial conditon 2')
plt.plot(simplex3[0,:],simplex3[1,:],label='initial conditon 3')


plt.plot(simplex1[0][0],simplex1[1][0],'.', ms=10, c='orange')
plt.plot(simplex2[0][0],simplex2[1][0],'.', ms=10, c='orange')
plt.plot(simplex3[0][0],simplex3[1][0],'.', ms=10, c='orange')


#plt.legend(loc =4, fontsize=10, labelspacing=0.01)
plt.plot(xax1,yax1,'-',c='black')
plt.plot(xax2,yax2,'-',c='black')
plt.plot(xax3,yax3,'-',c='black')

lab1=[0,1,0.5]
lab2=[0,0,0.8660254037844386]

lab3=['$x_{2}$','$x_{3}$','$x_{1}$']

#lab3=['$y_1$','$y_2$','$y_3$']

#lab3=['$z_1$','$z_2$','$z_3$']

ax = plt.gca()
ax.scatter(lab1, lab2)

for i, txt in enumerate(lab3):
    ax.annotate(txt, (lab1[i],lab2[i]) )

ax1 = plt.gca()
ax1.axes.get_xaxis().set_ticks([])
ax1.axes.get_yaxis().set_ticks([])
#frame.patch.set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#frame.axes.get_xaxis().set_visible(False)
plt.xlim([0.0,1.0])
plt.axes().set_aspect('equal', 'datalim')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#plt.axis('off')
#plt.savefig('fig8.png',dpi=3000)
plt.title("SGD 1")
plt.show()


#SGD 2 

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
plt.title("SGD 2", loc='left')
plt.show()    




