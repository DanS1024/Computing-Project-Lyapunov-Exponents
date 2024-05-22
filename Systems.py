import sympy as sp

discrete, continuous = [], []
class System:
    def __init__(self, name, dc, f, var, param, var0=None, param0=None, J=None):
        self.name = name
        self.dc = dc # discrete or continuous
        self.f = f # state equation
        self.var = var # variable names
        self.param = param # parameter names
        self.var0 = var0 # initial variable values
        self.param0 = param0 # initial parameter values
        self.J = J # Jacobian matrix

        if self.var0 == None:
            self.var0 = [0.1]*len(var)
        if self.param0 == None:
            self.param0 = [1]*len(param)
        if self.J == None:
            self.J = sp.Matrix(f).jacobian(var) # symbolic calculation of the Jacobian matrix
        
        if self.dc == 'd':
            discrete.append(self)
        elif self.dc == 'c':
            continuous.append(self)

# sympy symbols used
x, y, z, w, t, a, b, c, d, e, k, r = sp.symbols('x y z w t a b c d e k r', real=True)

# Logistic = System('Logistic Map', 'd', [r*x*(1-x)], [x], [r], [0.1], [4])

Tinkerbell = System('Tinkerbell Map', 'd', [x**2 - y**2 + a*x + b*y, 2*x*y + c*x + d*y], [x, y], [a, b, c, d], [0, 0.5], [0.9, -0.6, 2, 0.5])

PredatorPrey = System('Predator-Prey Map', 'd', [x*sp.exp(r*(1-x/k) - a*y), x*(1 - sp.exp(-a*y))], [x, y], [r, k, a], [0.5, 0.5], [3, 1, 5])

Henon = System('Henon Map', 'd', [1 - a*x**2 + y, b*x], [x, y], [a, b], [0.1, 0.1], [1.4, 0.3])

Lorenz3D = System('Lorenz 3D Map', 'd', [x*y - z, x, y], [x, y, z], [], [0.5, 0.5, -1.0])



Lorenz = System('Lorenz Attractor', 'c', [a*(y-x), x*(b-z)-y, x*y-c*z], [x, y, z], [a, b, c], [0, -0.01, 9], [10, 28, 8/3])

Pendulum = System('Damped Driven Pendulum', 'c', [1, y, -sp.sin(x) - b*y + a*sp.sin(w*t)], [t, x, y], [b, a, w], [0, 0, 0], [0.05, 0.6, 0.7])

Rossler = System('Rossler Attractor', 'c', [-y-z, x+a*y, b+z*(x-c)], [x, y, z], [a, b, c], [-9, 0, 0], [0.2, 0.2, 5.7])

Rossler4D = System('Hyperchaotic Rossler Attractor', 'c', [-y-z, x+a*y+w, b+x*z, d*w-c*z], [x, y, z, w], [a, b, c, d], [-10, -6, 0, 10], [0.25, 3, 0.5, 0.05])