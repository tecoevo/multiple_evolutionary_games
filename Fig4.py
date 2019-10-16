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
    
    d2 = 3
    D2 = d2 - 1
    
    for alpha1 in range (0,d1):
        for alpha2 in range (0,d1):
            for alpha3 in range (0,d1):
                if (alpha1+alpha2+alpha3== D1):
                    binom = comb1(D1,alpha1,alpha2,alpha3)
                    P_alpha = np.power(P[0][0],alpha1) * np.power(P[0][1],alpha2) * np.power(P[0][2],alpha3)
                    if (alpha2==0 and alpha3==0):
                        index = 0
                    elif (alpha2==1):
                          index = 1 
                    elif (alpha2==0 and alpha3==1):
                          index = 2    
                    F[0][0] = F[0][0] + ( binom * P_alpha * a1[0][index])
                    F[0][1] = F[0][1] + ( binom * P_alpha * a1[1][index])
                    F[0][2] = F[0][2] + ( binom * P_alpha * a1[2][index])
    #for alpha1 in range (0,d1):
        #for alpha2 in range (0,d1):
            #for alpha3 in range (0,d1):
                 #if (alpha1+alpha2+alpha3 == D1):
                     #binom = comb1(D1,alpha1,alpha2,alpha3)
                     #P_alpha = np.power(P[0][0],alpha1) * np.power(P[0][1],alpha2) * np.power(P[0][2],alpha3)
                     #if (alpha2==0 and alpha3==0):
                         #index = 0
                     #elif (alpha2==1):
                        #index = 1 
                     #elif (alpha2==0 and alpha3==1):
                          #index = 2    
                         # F[0][1] = F[0][1] + ( binom * P_alpha * a1[1][index])
                
                
    #for alpha1 in range (0,d1):
        #for alpha2 in range (0,d1):
            #for alpha3 in range (0,d1):
               #if (alpha1+alpha2+alpha3 == D1):
                    #binom = comb1(D1,alpha1,alpha2,alpha3)
                    #P_alpha = np.power(P[0][0],alpha1) * np.power(P[0][1],alpha2) * np.power(P[0][2],alpha3)
                   # if (alpha2==0 and alpha3==0):
                       # index = 0
                   # elif (alpha2==1):
                        #  index = 1 
                   # elif (alpha2==0 and alpha3==1):
                          #index = 2    
                         # F[0][2] = F[0][2] + ( binom * P_alpha * a1[2][index])
    
    
    
    
    
                
    for alpha1 in range (0,d2):
        for alpha2 in range (0,d2):
            if (alpha1+alpha2 == D2):
                binom = comb2(D2,alpha1,alpha2)
                P_alpha = np.power(P[1][0],alpha1) * np.power(P[1][1],alpha2)
                F[1][0] = F[1][0] + ( binom * P_alpha * a2[0][alpha2] )
                F[1][1] = F[1][1] + ( binom * P_alpha * a2[1][alpha2] )
    
    
    
    #for alpha1 in range (0,d2):
        #for alpha2 in range (0,d2):
            #if (alpha1+alpha2 == D2):
                #binom = comb2(D2,alpha1,alpha2)
                #P_alpha = np.power(P[1][0],alpha1) * np.power(P[1][1],alpha2)
                #F[1][1] = F[1][1] + ( binom * P_alpha * a2[1][alpha2] )
                
    #print ('F is  ', F)                
    return F
    

def PHI(FF,PP):
    
    phix = np.zeros((2,3),dtype = float)
    
    phix[0][0] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1]) + (PP[0][2]*FF[0][2])
    phix[0][1] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1]) + (PP[0][2]*FF[0][2])
    phix[0][2] = (PP[0][0]*FF[0][0]) + (PP[0][1]*FF[0][1]) + (PP[0][2]*FF[0][2])
    phix[1][0] = (PP[1][0]*FF[1][0]) + (PP[1][1]*FF[1][1]) + (PP[1][2]*FF[1][2])   # FF[1][2] = 0 as Game 2 has no strategy of type 3. Therefore, the third term just goes to zero
    phix[1][1] = (PP[1][0]*FF[1][0]) + (PP[1][1]*FF[1][1]) + (PP[1][2]*FF[1][2])
    
    
    #print ('phix is', phix)
    return  phix


def func(y,t,A1,A2):
    
    #print(' y is ', y)
    
    xx = np.array([[y[0],y[1]], [y[2],y[3]], [y[4],y[5]]],dtype = float)
    
    pp = np.zeros((2,3),dtype = float)
    
    yy = np.zeros((3,2),dtype = float)
    
    
    pp[0][0] = xx[0][0] + xx[0][1]
    pp[0][1] = xx[1][0] + xx[1][1]
    pp[0][2] = xx[2][0] + xx[2][1]
    pp[1][0] = xx[0][0] + xx[1][0] + xx[2][0]
    pp[1][1] = xx[0][1] + xx[1][1] + xx[2][1]
    
    #print ('p is', pp)
    
    f = fitness(pp,A1,A2) # 2D array with four elements 
    
    phi = PHI(f,pp)   # 2D array with four elements. Ensure rows have same value as for f11 and f12 --> same pi 
    
                        
    yy[0][0] = xx[0][0] * ( (f[0][0] - phi[0][0])  + (f[1][0]-phi[1][0]) ) 
    yy[0][1] = xx[0][1] * ( (f[0][0] - phi[0][0])  + (f[1][1]-phi[1][1]) ) 
    yy[1][0] = xx[1][0] * ( (f[0][1] - phi[0][1])  + (f[1][0]-phi[1][0]) ) 
    yy[1][1] = xx[1][1] * ( (f[0][1] - phi[0][1])  + (f[1][1]-phi[1][1]) ) 
    yy[2][0] = xx[2][0] * ( (f[0][2] - phi[0][2])  + (f[1][0]-phi[1][0]) ) 
    yy[2][1] = xx[2][1] * ( (f[0][2] - phi[0][2])  + (f[1][1]-phi[1][1]) ) 
    
    X11 = yy[0][0]
    X12 = yy[0][1]
    X21 = yy[1][0]
    X22 = yy[1][1]
    X31 = yy[2][0]
    X32 = yy[2][1]
    
    return X11,X12,X21,X22,X31,X32
    
def coordinates(a,b,c):
    
    Xnew = 1 - (a/2) - b
    Ynew = np.sqrt(3)/2 * a
    
    XN = np.array([Xnew],dtype = float)
    YN = np.array([Ynew],dtype = float)
    
    ret = np.concatenate([XN,YN])
    
    return ret
    
####### main #####

A_1 = np.array([[0, -1.0, 2.0], [2.0, 0 ,-1.0], [-1.0, 2.0, 0.0] ],dtype = float) # payoff matrices A 
A_2 = np.array([[10, 1, 5.5],[4, 10, 3]], dtype = float) # payoff matrices B


t = np.arange(0,60,0.01 )  # time 

l = len(t)



####### initial values for x11, x12, x21, x22, x31 and x32




initial1 = np.array([ 0.30 ,0.10 ,0.10 , 0.05, 0.40, 0.05 ],dtype = float)

#initial1 = np.array([ 0.4 ,0.0 ,0.1 , 0.3, 0.0, 0.2 ],dtype = float)

initial2 = np.array([ 0.4 ,0.1 ,0.2 , 0.1, 0.1, 0.1 ],dtype = float)

initial3 = np.array([ 0.2 ,0.3 ,0.1 , 0.1, 0.2, 0.1],dtype = float)


sol1  = odeint(func,initial1, t, args= (A_1,A_2)) 

sol2  = odeint(func,initial2, t, args= (A_1,A_2)) 

sol3  = odeint(func,initial3, t, args= (A_1,A_2))


x11_1 = sol1[:,0]
x12_1 = sol1[:,1]
x21_1 = sol1[:,2]
x22_1 = sol1[:,3]
x31_1 = sol1[:,4]
x32_1 = sol1[:,5]

p11_1 = x11_1 + x12_1
p12_1 = x21_1 + x22_1
p13_1 = x31_1 + x32_1
p21_1 = x11_1 + x21_1 + x31_1
p22_1 = x12_1 + x22_1 + x32_1


x11_2 = sol2[:,0]
x12_2 = sol2[:,1]
x21_2 = sol2[:,2]
x22_2 = sol2[:,3]
x31_2 = sol2[:,4]
x32_2 = sol2[:,5]

p11_2 = x11_2 + x12_2
p12_2 = x21_2 + x22_2
p13_2 = x31_2 + x32_2
p21_2 = x11_2 + x21_2 + x31_2
p22_2 = x12_2 + x22_2 + x32_2


x11_3 = sol3[:,0]
x12_3 = sol3[:,1]
x21_3 = sol3[:,2]
x22_3 = sol3[:,3]
x31_3 = sol3[:,4]
x32_3 = sol3[:,5]

p11_3 = x11_3 + x12_3
p12_3 = x21_3 + x22_3
p13_3 = x31_3 + x32_3
p21_3 = x11_3 + x21_3 + x31_3
p22_3 = x12_3 + x22_3 + x32_3

simplex1 = coordinates(p11_1,p12_1,p13_1)
simplex2 = coordinates(p11_2,p12_2,p13_2)
simplex3 = coordinates(p11_3,p12_3,p13_3)

equi1 = 0.127 * np.ones(l)
equi2 = 0.740 * np.ones(l)

equi3 = (1 - 0.127) * np.ones(l)
equi4 = (1- 0.740) * np.ones(l)



fig =plt.figure()
plt.rcParams.update({'font.size': 30, 'text.color':'black'})

plt.plot(simplex1[0,:],simplex1[1,:],linestyle='-.',c='blue',ms = 30,label='initial condition 1' ,linewidth=3.0)
plt.plot(simplex2[0,:],simplex2[1,:],linestyle='--',c='green',ms = 30,label='initial condition 2',linewidth=2.0)
plt.plot(simplex3[0,:],simplex3[1,:],linestyle='-',c='red',ms = 30,label='initial condition 3',linewidth=2.0)

plt.plot(simplex1[0][0],simplex1[1][0],'*', ms=8, markeredgewidth=0.0, c='black')
plt.plot(simplex2[0][0],simplex2[1][0],'*', ms=8, markeredgewidth=0.0, c='k')
plt.plot(simplex3[0][0],simplex3[1][0],'*', ms=8, markeredgewidth=0.0, c='k')
plt.plot(simplex3[0][5999],simplex3[1][5999],'.', ms=10, c='black')
#plt.legend(loc =4, fontsize=10, labelspacing=0.01)


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
plt.title("MGD, simplex a")
plt.savefig('../vandana/multiple/Omnigraffle_figures/3x3+2x2x2_a.svg',dpi=300)
plt.show()

plt.figure()
rcpars={ 'font.size': 25}# ,'figure.figsize': (3,2), 'figure.subplot.bottom': 0.175, 'figure.subplot.top': 0.9, 'figure.subplot.left': 0.2, 'savefig.dpi': 400}
plt.rcParams.update( rcpars)
plt.plot(t,p21_1,linestyle = '-.',c='blue',ms = 30,label='initial condition 1',linewidth=3.0)
plt.plot(t,p21_2,linestyle = '--',c='green',ms = 30,label='initial condition 2',linewidth=3.0)
plt.plot(t,p21_3,linestyle = '-',c='red',ms = 30,label='initial condition 3',linewidth=3.0)
plt.ylim((-0.1,1.1))
plt.xlim((-0.4,20.45))
plt.ylabel('$p_{21}$',size=23)
#plt.xlabel('time',size=20)
plt.plot(t,equi1,'k--',linewidth=2.0)
plt.plot(t,equi2,'k--',linewidth=2.0)
plt.title("MGD, simplex b, dashed lines corresponds to the two SGD equilibriums")
plt.savefig('../vandana/multiple/Omnigraffle_figures/3x3+2x2x2_b.svg',dpi=300)
plt.show()


fig = plt.figure()
ax = plt.subplot(111)
plt.plot(t,x11_1,label='x11_1')
plt.plot(t,x12_1,label='x12_1')
plt.plot(t,x21_1,label='x21_1')
plt.plot(t,x22_1,label='x22_1')
plt.plot(t,x31_1,label='x31_1')
plt.plot(t,x32_1,label='x32_1')
plt.ylim([0.0,1.0])
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.50),
          ncol=3, fancybox=True, shadow=True)
plt.ylim((-0.1,1.1))
plt.xlim((-0.4,30.45))    
plt.ylabel("freq of categorical type")
plt.xlabel("time")
plt.title("MGD")
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


#A = np.array([[0,1,-1],[-1,0,1],[1,-1,0]])

A = np.array([[0,-1,2],[2,0,-1],[-1,2,0]])

#A = np.array([[0,-3,1],[1,0,-3],[-3,1,0]])

#A = np.array([[-1,-5,5],[5,-1,-5],[-5,5,-1]])

#A = np.array([[-1,10,-10],[-6,-1,6],[2,-2,-1]])

#A = np.array([[-0.1,1,-1],[-1,-0.1,1],[1,-1,-0.1]])

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
def f1(x,A):
    
    pi_A = ( (x**2) * A[0][0] ) + (( 2 * x) * (( 1- x) * A[0][1]) )  + ((( 1 -x) **2) * A[0][2])  
    
    return pi_A


def f2(x,A):
    
    pi_B = ( (x**2) * A[1][0] ) + (( 2 * x) * (( 1-x) * A[1][1]) )  + ((( 1 - x) **2) * A[1][2])
    
    return pi_B
 
#payoff matrix 


A7 = np.array([[10,1,5.5],[4,10,3]])

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

equi1 = equ * 0.740
equi2 = equ * 0.127


ax1 = fig.gca()
plt.plot(x7,y7)
plt.plot(x7,zer,'r-')
plt.plot(equi1,y33,'k:')
plt.plot(equi2,y33,'k:')
plt.xlabel('$x_1$ (frequency of strategy 1)')
plt.ylabel("y = x' (time evolution of x)")
ax1.axes.get_xaxis().set_ticks([])
plt.title("SGD 2")
plt.show()







