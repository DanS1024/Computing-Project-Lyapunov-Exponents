# Python program to calculate the spectrum of Lyapunov exponents for discrete and continuous-time dynamical systems
import numpy as np
import sympy as sp
import Systems
from LCEs import *
from plotter import *
rng = np.random.default_rng()

m = 10**4 # number of transient iterations
n = 10**5 # number of iterations
dt = 0.005 # time step for continuous systems

trials = 10 # number of trials
dx = 0.01 # perturbation of initial conditions

# list of maps to run
Maps = Systems.discrete + Systems.continuous

showResults = True
saveResults = False
folder = '' # 'folder/'

for Map in Maps:
    name = Map.name
    dc = Map.dc
    f = Map.f
    J = Map.J
    var = Map.var
    param = Map.param
    param0 = Map.param0

    # make callable functions
    f_func = sp.lambdify([var, param], sp.Array(f))
    J_func = sp.lambdify([var, param], J)

    LCEs = [] # store final exponent of each run
    # run trials with different initial conditions
    for i in range(trials):
        # random perturbation of initial conditions
        var0 = Map.var0 + rng.uniform(-dx, dx, len(Map.var0))
        
        if dc == 'd': # discrete
            res = LyapunovDiscrete(f_func, J_func, var0, param0, m, n)
        elif dc == 'c': # continuous
            res = LyapunovContinuous(f_func, J_func, var0, param0, m, n, dt)
        # res format is [LCEs evolution, phase space trajectory]
        
        LCEs.append(res[0].T[-1]) # format is [lambda1, lambda2, ...]

    mean = np.mean(LCEs, axis=0)
    std = np.std(LCEs, axis=0, ddof=min(trials-1, 1)) / np.sqrt(trials) # standard deviation in the mean, ddof=1 for unbiased estimator

    # calculate the Kaplan-Yorke dimension
    k = 0
    tot = 0
    while k < len(mean) and tot + mean[k] + std[k] >= 0:
        tot += mean[k]
        k += 1
    dimKY = k + (0 if k == len(mean) else tot / abs(mean[k]))

    res += [mean, std, dimKY] # append other results

    plot(name, res, var, saveResults, folder)

if showResults: plt.show()