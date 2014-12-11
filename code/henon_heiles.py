import numpy as np
import matplotlib.pyplot as plt

def force(pos):
    x, y = pos
    return np.array([-x - 2 * x * y, -y - x * x + y * y])

def integration_step(pos, vel, dt):
    return pos + vel * dt, vel + force(pos) * dt

def integrate(pos0, vel0, times):
    ntimes = len(times)
    poss = np.zeros([2, ntimes])
    vels = np.zeros([2, ntimes])
    poss[0] = pos0
    vels[0] = vel0
    for ii in range(ntimes-1):
        p, v = integration_step(poss[ii], vels[ii], times[ii+1] - times[ii])
        poss[ii+1] = p
        vels[ii+1] = v
    return poss, vels

if __name__ == "__main__":
    dt = 0.001
    times = np.arange(0., 3., dt)
    pos0 = np.array([0.01, 0.95])
    vel0 = np.zeros_like(pos0)
    poss, vels = integrate(pos0, vel0, times)
    plt.clf()
    plt.plot(poss[0], poss[1])
    plt.savefig("hh.png")

