import matplotlib.pyplot as plt
import numpy as np
import math 
import random


EPSILON = 0.000000000000001

U0 = 0

h = 0.001

lambda_1 = 0
lambda_2 = 0
lambda_3 = 0.5
lambda_4 = 1

lambda_5 = 0
lambda_6 = 0
lambda_7 = 0.5
lambda_8 = 1

T1 = 0
T2 = 2
T = T2-T1
N=int(T/h)


def EE_infinit_prec(U, U0, lambda_):
    U[0] = U0
    for n in range(1, N):
        tn = n*h
        U[n] = U[n-1] + h*2*math.sqrt(abs(U[n-1]))*(tn > lambda_)
        
def EE_finit_prec(U, U0, lambda_):
    delta = EPSILON*random.uniform(0.0, 10.0)
    U[0] = U0+h*delta
    for n in range(1, N):
        tn = n*h
        U[n] = U[n-1] + h*2*math.sqrt(abs(U[n-1]))*(tn > lambda_)
        delta = EPSILON*random.uniform(0.0, 10.0)
        U[n] += h*delta

def EI_infinit_prec(U, U0, lambda_):
    U[0] = U0
    for n in range(1, N):
        tn = n*h
        U[n] = U[n-1] + 2*h*(h+math.sqrt(h*h+U[n-1]))*(tn > lambda_)

def EI_finit_prec(U, U0, lambda_):
    delta = EPSILON*random.uniform(0.0, 10.0)
    U[0] = U0+h*delta
    for n in range(1, N):
        tn = n*h
        U[n] = U[n-1] + 2*h*(h+math.sqrt(h*h+U[n-1]))*(tn > lambda_)
        delta = EPSILON*random.uniform(0.0, 10.0)
        U[n] += h*delta


def CN_infinit_prec(U, U0, lambda_):
    U[0] = U0
    for n in range(1, N):
        tn = n*h
        U[n] = (h + math.sqrt(U[n-1]))**2 * (tn > lambda_)


def CN_finit_prec(U, U0, lambda_):
    delta = EPSILON*random.uniform(0.0, 10.0)
    U[0] = U0+h*delta
    for n in range(1, N):
        tn = n*h
        delta = EPSILON*random.uniform(0.0, 10.0)
        U[n] = ((h + math.sqrt((h+2*math.sqrt(U[n-1]))**2 + 4*h*delta))/2)**2
        U[n] = U[n]*(tn > lambda_)


def H_infinit_prec(U, U0, lambda_):
    U[0] = U0
    for n in range(1, N):
        tn = n*h
        U[n] = U[n-1] + h*(math.sqrt(U[n-1]) + math.sqrt(U[n-1] + h*math.sqrt(U[n-1])))
        U[n] = U[n]*(tn > lambda_)


def H_finit_prec(U, U0, lambda_):
    delta = EPSILON*random.uniform(0.0, 10.0)
    U[0] = U0+h*delta
    for n in range(1, N):
        tn = n*h
        delta = EPSILON*random.uniform(0.0, 10.0)
        U[n] = U[n-1] + h*(math.sqrt(U[n-1]) + math.sqrt(U[n-1] + h*math.sqrt(U[n-1]))) + h*delta
        U[n] = U[n]*(tn > lambda_)


x = np.arange(T1, T2, h)

U_1 = np.empty(N)
U_2 = np.empty(N)
U_3 = np.empty(N)
U_4 = np.empty(N)

U_5 = np.empty(N)
U_6 = np.empty(N)
U_7 = np.empty(N)
U_8 = np.empty(N)

EE_infinit_prec(U_1, U0, lambda_1)
EE_finit_prec(U_2, U0, lambda_2)
#EE_finit_prec(U_3, U0, lambda_3)
#EE_finit_prec(U_4, U0, lambda_4)

H_finit_prec(U_5, U0, lambda_5)
#EI_finit_prec(U_6, U0, lambda_6)
#EI_finit_prec(U_7, U0, lambda_7)
#EI_finit_prec(U_8, U0, lambda_8)

#EI(U)
#for n in range(1, 2000):
#    tn = n*h
#    U[n] = U[n-1] + h*f_EE(tn, U[n-1])


plt.plot(x, U_1)
plt.plot(x, U_2)
#plt.plot(x, U_3)
#plt.plot(x, U_4)

#plt.plot(x, U_5)
#plt.plot(x, U_6)
#plt.plot(x, U_7)
#plt.plot(x, U_8)

plt.xlim(0, 2.5)
#plt.ylim(0, 5)
plt.grid(True)
plt.show()
