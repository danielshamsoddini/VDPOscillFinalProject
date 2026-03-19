import numpy as np

#Simple forward euler method for solving func from time_start to time_end with step dt and initial condition z_0
def euler(func, time_start, time_end, dt, z_0):
    time_interval = np.arange(time_start, time_end + 1e-12, dt)
    z_states = np.zeros((len(time_interval), len(z_0)), dtype=float)
    z_states[0] = z_0
    for a in range(len(time_interval) - 1):
        z_states[a+1] = z_states[a] + dt*func(time_interval[a], z_states[a])
    return time_interval, z_states

#RK4 method for solving func from time_start to time_end with step dt and initial condition z_0
def rk4(func, time_start, time_end, dt, z_0):
    time_interval = np.arange(time_start, time_end + 1e-12, dt)
    z_states = np.zeros((len(time_interval), len(z_0)), dtype=float)
    
    z_states[0] = z_0
    for a in range(len(time_interval) - 1):
        ti, zi = time_interval[a], z_states[a]
        k1 = func(ti, zi)
        k2 = func(ti + 0.5*dt, zi + 0.5*dt*k1)
        k3 = func(ti + 0.5*dt, zi + 0.5*dt*k2)
        k4 = func(ti + dt,     zi + dt*k3)
        z_states[a+1] = zi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return time_interval, z_states
