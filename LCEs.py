import numpy as np

def LyapunovDiscrete(f, J, var0, param0, m, n):
    x = f(var0, param0)
    # initial transient of m iterations
    for i in range(m):
        x = f(x, param0)

    # initial QR decomposition
    Q, R = np.linalg.qr(J(x, param0))
    
    traj = [] # trajectory for plotting
    Rdiag = []
    # iterate n times
    for i in range(n):
        x = f(x, param0)
        if n - i <= 10**4: traj.append(x) # append last iterations
        # QR decomposition of J*
        Q, R = np.linalg.qr(J(x, param0) @ Q)
        # save the diagonal elements of R
        Rdiag.append(np.diag(R))

    # spectrum of Lyapunov exponents (cumulative average)
    res = np.cumsum(np.log(np.abs(Rdiag)), axis=0).T / np.arange(1, n+1)
    return [res, np.array(traj).T]



# define different numerical integration methods
# evolves x and Q (Phi) at the same time for following the system
# dx/dt = f(x, param)
# dQ/dt = J(x, param) @ Q

def Euler(f, J, x, Q, param, dt):
    return x + dt*f(x, param), Q + dt*J(x, param) @ Q

def RK2(f, J, x, Q, param, dt):
    k1 = f(x, param)
    k1J = J(x, param) @ Q
    k2 = f(x + dt/2*k1, param)
    k2J = J(x + dt/2*k1, param) @ (Q + dt/2*k1J)
    return x + dt*k2, Q + dt*k2J

def RK3(f, J, x, Q, param, dt):
    k1 = f(x, param)
    k1J = J(x, param) @ Q
    k2 = f(x + dt/2*k1, param)
    k2J = J(x + dt/2*k1, param) @ (Q + dt/2*k1J)
    k3 = f(x - dt*k1 + 2*dt*k2, param)
    k3J = J(x - dt*k1 + 2*dt*k2, param) @ (Q - dt*k1J + 2*dt*k2J)
    return x + dt/6*(k1 + 4*k2 + k3), Q + dt/6*(k1J + 4*k2J + k3J)

def RK4(f, J, x, Q, param, dt):
    k1 = f(x, param)
    k1J = J(x, param) @ Q
    k2 = f(x + dt/2*k1, param)
    k2J = J(x + dt/2*k1, param) @ (Q + dt/2*k1J)
    k3 = f(x + dt/2*k2, param)
    k3J = J(x + dt/2*k2, param) @ (Q + dt/2*k2J)
    k4 = f(x + dt*k3, param)
    k4J = J(x + dt*k3, param) @ (Q + dt*k3J)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4), Q + dt/6*(k1J + 2*k2J + 2*k3J + k4J)

def LyapunovContinuous(f, J, var0, param0, m, n, dt, order=4):
    N = len(var0)
    func = [Euler, RK2, RK3, RK4][order-1] # choose method
    x, Q = var0, np.eye(N) # initial conditions

    traj = []
    Rdiag = []
    for i in range(m+n):
        x, Q = func(f, J, x, Q, param0, dt) # integrate system
        if m+n - i <= 10**5: traj.append(x) # append last iterations
        Q, R = np.linalg.qr(Q) # QR decomposition of Q (Phi)
        if i >= m: Rdiag.append(np.diag(R)) # save diagonal of R

    # spectrum of Lyapunov exponents (cumulative average)
    res = np.cumsum(np.log(np.abs(Rdiag))/dt, axis=0).T / np.arange(1, n+1)
    return [res, np.array(traj).T]

