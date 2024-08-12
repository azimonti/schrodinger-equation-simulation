#!/usr/bin/env python3
'''
/************************/
/*  mod_config_1d.py    */
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
    potential=0,  # potential to use
    omega=1.0,    # angular frequency of the harmonic oscillator
    Vx_bar=5.0,   # barrier location
    V_barrier=2.0,  # infinite long barrier potential
    Vx_finite_bar=6.2  # finite barrier end
)

electron_params = SimpleNamespace(
    x0=1.5e-10,       # initial position of the wavepacket in meters
    p=1.0e-24,        # initial momentum of the wavepacket in kg*m/s
    sigma=1e-10,      # sigma of the wavepacket in meters
    hbar=1.054e-34,   # Planck's constant (scaled)
    m=9.109e-31,      # mass of an electron in kg
    dx=1e-12,         # spatial step size in meters
    dt=1e-18,         # time step size in seconds
    x_max=30e-10,     # spatial domain limit in meters
    t_max=9e-16,      # time domain limit in seconds
    potential=0,      # potential to use
    omega=4.5e15,     # angular frequency of the harmonic oscillator in rad/s
    Vx_bar=5e-10,     # barrier location in meters
    V_barrier=2e-18,  # infinite long barrier potential in joules
    Vx_finite_bar=6.2e-10  # finite barrier end in meters
)

cfg = SimpleNamespace(
    small_scale=True,     # domain scaled at electron size
    superposition=False,  # use wavepackets or eigenfunctions
    eigenfunctions_list=np.array([1, 2]),  # superposition list
    compute_prob=False,   # compute probability at each step
    plot_prob=True,       # plot probability or wavefunction
    plot_phase=True,      # plot real an imaginary or phase
    increase_precision=True,  # use Runge-Kutta or Euler
    rk_method='DOP853',   # RK method 'RK23', 'RK45', or 'DOP853'
    high_res=True,        # enable high resolution simulation
    plot=True,            # enable plotting
    compute=True,         # enable computations
    animate=True,         # enable animations
    load_data=False,      # load data from a file
    save_data=False,      # save data to a file
    data_folder='data/simul',  # folder for data files
    animation_format='mp4',    # format for animations
    total_duration=6,     # total duration of the simulation
    fps=30,               # frames per second for the animation
    verbose=False
)

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    pass
