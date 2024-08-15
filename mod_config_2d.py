#!/usr/bin/env python3
'''
/************************/
/*  mod_config_2d.py    */
/*     Version 1.0      */
/*      2024/08/12      */
/************************/
'''
import sys
from types import SimpleNamespace

p2 = SimpleNamespace(
    # low resolution
    Nx_lr=200,             # number of grid points in x direction
    Ny_lr=200,             # number of grid points in y direction
    dt_lr=0.01,            # time step for the simulation
    # high resolution
    Nx_hr=400,             # number of grid points in x direction
    Ny_hr=400,             # number of grid points in y direction
    dt_hr=0.005,           # time step for the simulation
    x_min=-20,             # minimum x value
    x_max=20,              # maximum x value
    y_min=-15,             # minimum y value
    y_max=15,              # maximum y value
    x0=-15,                # initial position of the wave packet in x
    y0=0,                  # initial position of the wave packet in y
    sigma_x=2.0,           # width of the wave packet in x direction
    sigma_y=2.0,           # width of the wave packet in y direction
    kx=50.0,               # initial wave vector in x direction
    ky=0.0,                # initial wave vector in y direction
    t_max=15,              # maximum simulation time
    barrier_height=47.0,   # potential of the finite barrier
    barrier_center= 0.0,   # center of the finite barrier
    barrier_width=0.4,     # half width in the x direction
    total_duration=13,     # total duration of the animation
    fps=30                 # frames per second for the animation
)


cfg = SimpleNamespace(
    plot_prob=True,           # plot probability or wavefunction
    dev_simul=False,          # align the number of steps to number of frames
    infinite_barrier=True,    # reflecting or absorbing boundary
    middle_barrier=True,      # set a finite barrier at x/2
    periodic_boundary=False,  # periodic boundary
    high_res_grid=False,      # enable high resolution simulation grid
    high_res_dt=False,        # enable high resolution simulation timestep
    high_res_plot=True,       # enable high resolution simulation plot
    plot=False,               # enable plotting
    compute=True,             # enable computations
    animate=True,             # enable animations
    save_anim=True,           # save animations
    load_data=False,          # load data from a file
    save_data=True,           # save data to a file
    data_folder='data/simul',  # folder for data files
    animation_format='mp4',   # animation format (mp4 or gif)
    verbose=True
)

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    pass
