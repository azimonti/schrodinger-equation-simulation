#!/usr/bin/env python3
'''
/************************/
/*     mod_config.py    */
/*     Version 1.0      */
/*      2024/06/02      */
/************************/
'''
import numpy as np
import sys
from types import SimpleNamespace

p1 = SimpleNamespace(
    x0=2.0,       # initial position of the wavepacket
    p=1.0,        # initial momentum of the wavepacket
    sigma=1.0,    # sigma of the wavepacket
    hbar=1.0,     # Planck's constant (scaled)
    m=1.0,        # mass of particles
    dx=0.01,      # spatial step size
    dt=0.01,      # time step size
    x_max=30.0,   # spatial domain limit
    t_max=9.0,    # time domain limit
    potential=3,  # potential to use
    omega=1.0,    # angular frequency of the harmonic oscillator
    Vx_bar=5.0,   # barrier location
    V_barrier=2.0,  # infinite long barrier potential
    Vx_finite_bar=6.2  # finite barrier end
)

cfg = SimpleNamespace(
    superposition=False,  # use wavepackets or eigenfunctions
    eigenfunctions_list=np.array([1, 2]),  # superposition list
    compute_prob=True,  # compute probability at each step
    plot_prob=True,     # plot probability or wavefunction
    plot_phase=True,    # plot real an imaginary or phase
    increase_precision=True,  # use Runge-Kutta or Euler
    rk_method='DOP853'  # RK method 'RK23', 'RK45', or 'DOP853'
)

palette = SimpleNamespace(
    b=(102 / 255, 204 / 255, 255 / 255),  # blue
    o=(255 / 255, 153 / 255, 102 / 255),  # orange
    r=(204 / 255, 0 / 255, 102 / 255),    # red
    g=(102 / 255, 204 / 255, 102 / 255)   # green
)

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    pass
