import numpy as np
import matplotlib.pyplot as plt

def hamiltonian(pos, vel):
    return 0.5 * np.sum(vel * vel) + 0.5 * np.sum(pos * pos) + pos[0] * pos[0] * pos[1] - pos[1] * pos[1] * pos[1] / 3.

def force(pos):
    x, y = pos
    return np.array([-x - 2 * x * y, -y - x * x + y * y])

def integration_step(pos, vel, dt):
    """
    Note: secret leap-frog!
    """
    p = pos + vel * dt
    return p, vel + force(p) * dt

def integrate(pos0, vel0, times):
    ntimes = len(times)
    poss = np.zeros([2, ntimes])
    vels = np.zeros([2, ntimes])
    hs = np.zeros(ntimes)
    poss[:, 0] = pos0
    vels[:, 0] = vel0
    hs[0] = hamiltonian(pos0, vel0)
    for ii in range(ntimes-1):
        p, v = integration_step(poss[:, ii], vels[:, ii], times[ii+1] - times[ii])
        poss[:, ii+1] = p
        vels[:, ii+1] = v
        hs[ii+1] = hamiltonian(p, v)
    return poss, vels, hs

if __name__ == "__main__":
    dt = 0.0003
    times = np.arange(0., 50., dt)
    pos0 = np.array([0.1, 0.80])
    vel0 = np.array([-0.05, 0.02])
    poss, vels, hs = integrate(pos0, vel0, times)
    plt.figure(figsize=(6., 6.))
    plt.clf()
    alpha = 0.75
    plt.plot(times, hs, "k-", alpha=alpha)
    plt.xlabel("t")
    plt.ylabel("H")
    plt.ylim(np.array([0.99, 1.01]) * np.median(hs))
    plt.savefig("hh_H.png")
    plt.clf()
    plt.plot(poss[0, :], poss[1, :], "k-", alpha=alpha)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig("hh_pos.png")
    pos0 = pos0 + np.array([0.001, 0.])
    poss2, vels2, hs2 = integrate(pos0, vel0, times)
    plt.plot(poss2[0, :], poss2[1, :], "r-", alpha=alpha)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig("hh_pos_compare.png")
    plt.clf()
    plt.plot(times, np.sqrt(np.sum((poss - poss2) ** 2, axis=0)), "k-", alpha=alpha)
    plt.savefig("hh_deviation.png")
