import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import math 
import random

S0 = 1000000.0
I0 = 30.0
R0 = 0.0
SUM = S0+I0+R0
r = 0.5
a = 0.33
T0 = 0.0
T = 150.0
N = 150
h = float((T-T0)/N)

aB = [[ 0,   0,   0,   0 ], 
      [ 1/2, 0,   0,   0 ],
      [ 0,   1/2, 0,   0 ],
      [ 0,   0,   1,   0 ]]

bB =  [ 1/6, 1/3, 1/3, 1/6 ]


def fI(Sn, In):
    return 

def fR(In):
    return 

def fSIR(y):
    S, I, R = y
    dydt = np.array([-r * S * I / SUM, 
                     (r * S / SUM - a) * I, 
                     a * I])
    return dydt

def fSIR_func(y, t):
    dydt = fSIR(y)
    return dydt


def EE_infinit_prec(y):
    for n in range(1, N):
        y[n] = y[n-1] + h * fSIR(y[n-1])


def Heun_infinit_prec(y):
    for n in range(1, N):
        y[n] = y[n-1] + h/2 * (fSIR(y[n-1]) + fSIR(y[n-1] + h * fSIR(y[n-1])))


def Runge_Kutta_infinit_prec(y):
    K = np.empty(shape=(4,3))
    for n in range(1, N):
        K[0] = fSIR(y[n-1])
        K[1] = fSIR(y[n-1] + h * aB[1][0] * K[0])
        K[2] = fSIR(y[n-1] + h * (aB[2][0] * K[0] + aB[2][1] * K[1]))
        K[3] = fSIR(y[n-1] + h * (aB[3][0] * K[0] + aB[3][1] * K[1]) + aB[3][2] * K[2])

        y[n] = y[n-1] + h * (bB[0] * K[0] + bB[1] * K[1] + bB[2] * K[2] + bB[3] * K[3])


def solveSIR(y0, t):
    assert t.ndim == 1
    
    y = np.empty(shape=(t.shape[0], 3))
    y[0] = y0
    Runge_Kutta_infinit_prec(y)
    return y



t = np.arange(T0, T, h)
y0 = np.array([S0, I0, R0])

#sol = solveSIR(y0, t)
sol = odeint(fSIR_func, y0, t)

plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='R')

plt.legend(loc='best')
plt.xlabel('t')
plt.xlim(T0, T)
plt.ylim(-0.05 * 1e6, 1.05 * 1e6)
plt.grid(True)
plt.show()



