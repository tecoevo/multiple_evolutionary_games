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
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter




def comb(N,k1,k2):

    C = ( mt.factorial(N) / ( mt.factorial(k1) * mt.factorial(k2) ) )

    return C


def fitness(P,a1,a2):

    F = np.zeros((2,2),dtype = float)

    # number of players

    d1 = 2
    D1 = d1 - 1

    d2 = 3
    D2 = d2 - 1

    for alpha1 in range (0,d1):
        for alpha2 in range (0,d1):
            if (alpha1+alpha2 == D1):
                binom = comb(D1,alpha1,alpha2)
                P_alpha = np.power(P[0][0],alpha1) * np.power(P[0][1],alpha2)
                F[0][0] = F[0][0] + ( binom * P_alpha * a1[0][alpha2] )
                F[0][1] = F[0][1] + ( binom * P_alpha * a1[1][alpha2] )



    for alpha1 in range (0,d2):
        for alpha2 in range (0,d2):
            if (alpha1+alpha2 == D2):
                binom = comb(D2,alpha1,alpha2)
                P_alpha = np.power(P[1][0],alpha1) * np.power(P[1][1],alpha2)
                F[1][0] = F[1][0] + ( binom * P_alpha * a2[0][alpha2] )
                F[1][1] = F[1][1] + ( binom * P_alpha * a2[1][alpha2] )

    #print ('F is  ', F)
    return F


def PHI(FF,PP):

    phix = np.zeros((2,2),dtype = float)

    phix[0][0] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1])
    phix[0][1] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1])
    phix[1][0] = (PP[1][0]*FF[1][0]) + (PP[1][1]*FF[1][1])
    phix[1][1] = (PP[1][0]*FF[1][0]) + (PP[1][1]*FF[1][1])


    #print ('phix is', phix)
    return  phix


def func(y,t,A1,A2):

    #print(' y is ', y)

    xx = np.array([[y[0],y[1]], [y[2],y[3]]],dtype = float)

    pp = np.zeros((2,2),dtype = float)

    yy = np.zeros((2,2),dtype = float)


    pp[0][0] = xx[0][0] + xx[0][1]
    pp[0][1] = xx[1][0] + xx[1][1]
    pp[1][0] = xx[0][0] + xx[1][0]
    pp[1][1] = xx[0][1] + xx[1][1]

    #print ('p is', pp)

    f = fitness(pp,A1,A2) # 2D array with four elements

    phi = PHI(f,pp)   # 2D array with four elements. Ensure rows have same value as for f11 and f12 --> same pi


    yy[0][0] = xx[0][0] * ( (f[0][0] - phi[0][0])  + (f[1][0]-phi[1][0]) )
    yy[0][1] = xx[0][1] * ( (f[0][0] - phi[0][0])  + (f[1][1]-phi[1][1]) )
    yy[1][0] = xx[1][0] * ( (f[0][1] - phi[0][1])  + (f[1][0]-phi[1][0]) )
    yy[1][1] = xx[1][1] * ( (f[0][1] - phi[0][1])  + (f[1][1]-phi[1][1]) )

    X11 = yy[0][0]
    X12 = yy[0][1]
    X21 = yy[1][0]
    X22 = yy[1][1]

    return X11,X12,X21,X22

def coordinates(a,b,c,d):

    Xnew = (b) + ( 0.5 *( c+d) )
    Ynew = np.sqrt(3) * (c/2 + d/6)
    Znew =  np.sqrt(6) * d /3

    XN = np.array([Xnew],dtype = float)
    YN = np.array([Ynew],dtype = float)
    ZN = np.array([Znew],dtype = float)
    ret = np.concatenate([XN,YN,ZN])

    return ret

####### main #####

A_1 = np.array([[-1, 1], [0,0]],dtype = float) # payoff matrices A

#A_2 = np.array([[10, 1, 5.5],[4, 10, 3]], dtype = float) # payoff matrices B

A_2 = np.array([[-2,3, -2],[0,0,0]], dtype = float) # payoff matrices B


t = np.arange(0,100,0.01 )  # time

l = len(t)

x_11 = np.zeros(l)
x_12 = np.zeros(l)
x_21 = np.zeros(l)
x_22 = np.zeros(l)

####### initial values for x11, x12, x21 and x22

x110  = 0.2
x120=  0.1
x210= 0.2
x220=  1 - (x110 + x120 +x210)

initial1 = np.array([ x110 , x120, x210 , x220 ],dtype = float)

initial2 = np.array([ 0.1 ,0.1 ,0.6 , 0.2 ],dtype = float)

initial3 = np.array([ 0.1 ,0.6 ,0.1 , 0.2 ],dtype = float)

#p110 = x110 + x120
#p120 = x210 + x220
#p210 = x110 + x210
#p220 = x120 + x220

sol1  = odeint(func,initial1, t, args= (A_1,A_2))

sol2  = odeint(func,initial2, t, args= (A_1,A_2))

sol3  = odeint(func,initial3, t, args= (A_1,A_2))


x11_1 = sol1[:,0]
x12_1 = sol1[:,1]
x21_1 = sol1[:,2]
x22_1 = sol1[:,3]

p11_1 = x11_1 + x12_1
p12_1 = x21_1 + x22_1
p21_1 = x11_1 + x21_1
p22_1 = x12_1 + x22_1


x11_2 = sol2[:,0]
x12_2 = sol2[:,1]
x21_2 = sol2[:,2]
x22_2 = sol2[:,3]

p11_2 = x11_2 + x12_2
p12_2 = x21_2 + x22_2
p21_2 = x11_2 + x21_2
p22_2 = x12_2 + x22_2


x11_3 = sol3[:,0]
x12_3 = sol3[:,1]
x21_3 = sol3[:,2]
x22_3 = sol3[:,3]

p11_3 = x11_3 + x12_3
p12_3 = x21_3 + x22_3
p21_3 = x11_3 + x21_3
p22_3 = x12_3 + x22_3

simplex1 = coordinates(x22_1,x11_1,x12_1,x21_1)
simplex2 = coordinates(x22_2,x11_2,x12_2,x21_2)
simplex3 = coordinates(x22_3,x11_3,x12_3,x21_3)


T = np.arange(-0.1,0.2,0.1)

S = len(T)

K11 =  np.zeros(S)
K12 =  np.zeros(S)
K21 =  np.zeros(S)
K22 =  np.zeros(S)

K11= T + 0.3
K12= 0.2 - T
K21= 0.3 - T
K22= 0.2 + T


linex = coordinates(K22,K11,K12,K21)




fig = plt.figure()
fig = plt.figure(figsize=plt.figaspect(0.5)*1.0)
ax = fig.gca(projection='3d')
#ax.set_aspect('equal')

ax.plot(simplex1[0,:],simplex1[1,:],simplex1[2,:],c='blue',ms = 50,linestyle='-.',linewidth = 2.0,label='initial condition 1')
#ax.scatter(simplex1[0][0],simplex1[1][0],simplex1[2][0], marker ='*',s= 30,color ='orange')

ax.plot(simplex2[0,:],simplex2[1,:],simplex2[2,:],c='green',ms = 50,linestyle='--',linewidth = 2.0,label='initial condition 2')
#ax.scatter(simplex2[0][0],simplex2[1][0],simplex2[2][0],marker ='*',s= 30, color ='orange')

ax.plot(simplex3[0,:],simplex3[1,:],simplex3[2,:],c='red', ms = 50,linestyle='-',linewidth =2.0,label='initial condition 3')
#ax.scatter(simplex3[0][0],simplex3[1][0],simplex3[2][0], marker ='*',s= 30,color ='orange')

ax.text(0, 0, 0, "$x_{22}$", color='magenta',fontsize=15)
ax.text(1, 0, 0, "$x_{11}$", color='magenta',fontsize=15)
ax.text(0.5,0.8660254 , 0, "$x_{12}$", color='magenta',fontsize=15)
ax.text(0.5, 0.28867513,0.81649658 , "$x_{21}$", color='magenta',fontsize=15)

axx1 = np.array([0,1,0.5,0.5])
ayy2 = np.array([0,0,np.sqrt(3)/2,0.28867513 ])

azz3 = np.array([0,0,0,1.22474487])

xax1 = np.array([0.5,0])
yax1 = np.array([ 0.8660254,0  ])
zax1 = np.array([0,0])


xax2 = np.array([0.5,1])
yax2 = np.array([ 0.8660254,0  ])
zax2 = np.array([0,0])

xax3 = np.array([1,0])
yax3 = np.array([ 0.0,0.0  ])
zax3 = np.array([0,0])

xax4 = np.array([0.5,0.5])
yax4 = np.array([ 0.8660254, 0.28867513 ])
zax4 = np.array([0,0.81649658])

xax5 = np.array([0,0.5])
yax5 = np.array([0, 0.28867513 ])
zax5 = np.array([0,0.81649658])

xax6 = np.array([1,0.5])
yax6 = np.array([0, 0.28867513  ])
zax6 = np.array([0,0.81649658])

#ax.plot(axx1,ayy2,azz3,'-')

plt.plot(xax1,yax1,zax1,'-',c='black',linewidth=1.5)
plt.plot(xax2,yax2,zax2,'-',c='black',linewidth=1.5)
plt.plot(xax3,yax3,zax3,'-',c='black',linewidth=1.5)
plt.plot(xax4,yax4,zax4,'-',c='black',linewidth=1.5)
plt.plot(xax5,yax5,zax5,'-',c='black',linewidth=1.5)
plt.plot(xax6,yax6,zax6,'-',c='black',linewidth=1.5)

#plt.plot(linex[0,:],linex[1,:],linex[2,:],'-',c='red')

L = 10
h = 10


wright = np.zeros((L+1,3))


for ii in range(0,L+1):
    for jj in range(0,L+1):
        for kk in range(0,L+1):
            for ll in range(0,L+1):
                  #print ('x21 is', kk/h, '', 'x12 is', jj/h,'', 'x11 is', ii/h,'','x22 is', ll/h)
                  if(ll/h+ii/h+jj/h+kk/h == 1.0):
                      if( ii/h - ( ii/h*ii/h)- (ii/h*jj/h) -(ii/h*kk/h) -(jj/h*kk/h) == 0.0  ):
                        print('x11 is', ii/h,'','x12 is', jj/h,'','x21 is', kk/h,'','x22 is', ll/h)
                        wright[ll] = coordinates(ll/h,ii/h,jj/h,kk/h)



print ('wright is', wright)


#ax.plot_trisurf(wright[:,0],wright[:,1],wright[:,2] ,color = 'white', linewidth=0, antialiased=True)
#ax.plot_wireframe(wright,rstride=10, cstride=10)
#ax.scatter(wright[:,0],wright[:,1],wright[:,2],c='seashell')


x = wright[:,0]
y = wright[:,1]
z = wright[:,2]
xyz = {'x': x, 'y': y, 'z': z}

# put the data into a pandas DataFrame (this is what my data looks like)
df = pd.DataFrame(xyz, index=range(len(xyz['x'])))

# re-create the 2D-arrays
x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='linear')

#ax.plot_wireframe(x2, y2, z2, rstride=10, cstride=10, color ='lightgrey' , linewidth=0.1, antialiased=False)



for angle in range(0, 230):
    ax.view_init(30, angle)
    plt.draw()
#plt.legend(loc =4, fontsize=10, labelspacing=0.01)
#ax1 = plt.gca()
ax.axis('off')
#ax.axes.get_xaxis().set_ticks([])
#ax.axes.get_yaxis().set_ticks([])
#frame.patch.set_visible(False)
#frame.axes.get_yaxis().set_visible(False)
#frame.axes.get_xaxis().set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.title("MGD")
plt.savefig('../vandana/multiple/manuscript/tetrahedron.svg',dpi=300)
plt.show()

plt.figure()
rcpars={ 'font.size': 25}# ,'figure.figsize': (3,2), 'figure.subplot.bottom': 0.175, 'figure.subplot.top': 0.9, 'figure.subplot.left': 0.2, 'savefig.dpi': 400}
plt.rcParams.update( rcpars)
plt.plot(t,p21_1,linestyle = '-.',c='blue',ms = 30,label='initial condition 1',linewidth=3.0)
plt.plot(t,p21_2,linestyle = '--',c='green',ms = 30,label='initial condition 2',linewidth=3.0)
plt.plot(t,p21_3,linestyle = '-',c='red',ms = 30,label='initial condition 3',linewidth=3.0)
plt.ylim((-0.1,1.1))
plt.xlim((-0.4,20.4))
plt.ylabel('$p_{21}$',size=23)
plt.xlabel('time',size=20)
plt.title("MGDb")
plt.savefig('../vandana/multiple/manuscript/2x2+2x2x2_a.svg',dpi=300)
plt.show()

plt.figure()
#rcpars={ 'font.size': 25}# ,'figure.figsize': (3,2), 'figure.subplot.bottom': 0.175, 'figure.subplot.top': 0.9, 'figure.subplot.left': 0.2, 'savefig.dpi': 400}
#plt.rcParams.update( rcpars)
plt.plot(t,p11_1,linestyle = '-.',c='blue',ms = 30,label='initial condition 1',linewidth=3.0)
plt.plot(t,p11_2,linestyle = '--',c='green',ms = 30,label='initial condition 2',linewidth=3.0)
plt.plot(t,p11_3,linestyle = '-',c='red',ms = 30,label='initial condition 3',linewidth=3.0)
plt.ylim((-0.1,1.1))
plt.xlim((-0.4,20.4))
plt.ylabel('$p_{11}$',size=23)
plt.xlabel('time',size=20)
plt.title("MGDa")
plt.savefig('../vandana/multiple/manuscript/2x2+2x2x2_b.svg',dpi=300)
plt.show()


# SGD 1


#write payoff matrix elements values 


a5 = -1.0
b5= 1.0
c5 = 0
d5 = 0

h = 0.01 #step size

x1 = np.arange(0,1,h)

x2 = 1- x1

l = len(x1)

zer = np.zeros(l)


y33= np.arange(-0.5,0.5,0.1)

L = len(y33)

equ = np.ones(L)

equi = equ * 0.6
equi1 = equ * 0.5


y1 = np.zeros(l)

y2 = np.zeros(l)

y3 = np.zeros(l)

y4 = np.zeros(l)

y5 = np.zeros(l)

f1_5 = x1*a5 + x2*b5 

f2_5 = x1*c5 + x2*d5 

y5 = x1prime = x1 * (1-x1) *(f1_5-f2_5)



ax = plt.gca()
plt.plot(x1,y5)
plt.plot(x1,zer,'r-')
plt.plot(equi1,y33,'k:')
#plt.title('(c)   a > c and b < d')
#plt.ylim([-0.5,0.5])
#plt.xlim([-0.1,1.1])
#ax.set_xticklabels([])
#ax.set_yticklabels([])
plt.title("SGD 1")
plt.xlabel('x (frequency of strategy 1)')
plt.ylabel("x' ")
plt.xlabel('$x$  ')
plt.ylabel(" $\dot{x}$ ")
ax.set_yticklabels([])
plt.savefig('../vandana/multiple/figures/presentation_2x2+2x2x2_SGD2.svg',dpi=300)
plt.show()



#SGD 2 
def f1(x,A):
    
    pi_A = ( (x**2) * A[0][0] ) + (( 2 * x) * (( 1- x) * A[0][1]) )  + ((( 1 -x) **2) * A[0][2])  
    
    return pi_A


def f2(x,A):
    
    pi_B = ( (x**2) * A[1][0] ) + (( 2 * x) * (( 1-x) * A[1][1]) )  + ((( 1 - x) **2) * A[1][2])
    
    return pi_B
 
#payoff matrix 


A7 = np.array([[-2,3,-2],[0,0,0]])

# Initial conditions 

X0 = 0.1
Y0 =0.0

initialone = np.array([X0,Y0])




h = 0.001 #step size

x1 = np.arange(0,1,h)
l = len(x1)


x7 = np.arange(0,1,h)

y7 = x7 * (1-x7) *(f1(x7,A7) - f2(x7,A7))

zer = np.zeros(l)

y33= np.arange(-0.5,0.5,0.1)

L = len(y33)

equ = np.ones(L)

#equi1 = equ * 0.740
#equi2 = equ * 0.127

equi1 = equ * 0.7236068
equi2 = equ * 0.2763932


ax = fig.gca()
plt.plot(x7,y7)
plt.plot(x7,zer,'r-')
plt.plot(equi1,y33,'k:')
plt.plot(equi2,y33,'k:')
plt.title("SGD 2")
plt.xlabel('x (frequency of strategy 1)')
plt.ylabel("y = x' (time evolution of x)")
plt.xlabel('$x$ ')
plt.ylabel(" $\dot{x}$ ")
ax.axes.get_yaxis().set_visible(False)
plt.savefig('../vandana/multiple/figures/presentation_2x2+2x2x2_SGD1.svg',dpi=300)
plt.show()



