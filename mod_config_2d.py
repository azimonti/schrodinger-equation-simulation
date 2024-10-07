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
    Nx_lr=200,                # number of grid points in x direction
    Ny_lr=200,                # number of grid points in y direction
    dt_lr=0.01,               # time step for the simulation
    # high resolution
    Nx_hr=500,                # number of grid points in x direction
    Ny_hr=400,                # number of grid points in y direction
    dt_hr=0.005,              # time step for the simulation
    x_min=-30,                # minimum x value
    x_max=30,                 # maximum x value
    y_min=-20,                # minimum y value
    y_max=20,                 # maximum y value
    x0=-15,                   # initial position of the wave packet in x
    y0=0,                     # initial position of the wave packet in y
    sigma_x=3.0,              # width of the wave packet in x direction
    sigma_y=3.0,              # width of the wave packet in y direction
    kx=-1e6,                  # initial wave vector in x direction
    ky=0.0,                   # initial wave vector in y direction
    t_max=15,                 # maximum simulation time
    total_duration=15,        # total duration of the animation
    fps=30,                   # frames per second of the animation
    infinite_barrier=False,   # reflecting or absorbing boundary
    middle_barrier=True,      # set a finite barrier at barrier_center
    slits=False,              # allow horizontal holes in the barrier (slits)
    barrier_height=47.0,      # potential of the finite barrier
    barrier_center=0.0,       # center of the finite barrier
    barrier_width=0.4,        # half width in the x direction
    barriers_start=[15, 1, -2],  # start of  middle barriers
    barriers_end=[2, -1, -15],   # end of middle barriers
    periodic_boundary=False,  # periodic boundary
    absorbing_method=1,       # 0 (tanh), 1 CAP
    cap_type=0,               # 0 polynomial 1 optimal
    cap_poly=2,               # cap polynomial (e.g. 2 quadratic)
    cap_opt_a=2.62,           # a value for adaptive CAP
    absorbing_strength=20,    # absorbing strength
    absorbing_width_x=0.2,    # absorbing width x boundary
    absorbing_width_y=0.2,    # absorbing width y boundary
    absorbing_xmin=True,      # absorbing on left border
    absorbing_xmax=True,      # absorbing on right border
    absorbing_ymin=True,      # absorbing on lower border
    absorbing_ymax=True       # absorbing on upper border
)

p2_changes_load_s2d = SimpleNamespace()

cfg = SimpleNamespace(
    plot_prob=True,           # plot probability or wavefunction
    dev_simul=False,          # align the number of steps to number of frames
    plot_all_frames=False,    # plot all frames while computing
    display_all_d=True,       # display all domain if absorbing boundary
    show_smooth_d=True,       # show the absorbing smooth domain
    high_res_grid=False,      # enable high resolution simulation grid
    high_res_dt=False,        # enable high resolution simulation timestep
    high_res_plot=True,       # enable high resolution simulation plot
    fig_4k=True,              # use 4k resolution
    fix_min_max=True,         # fix the min max in the z direction
    z_xmax_scale=1,           # scale the max
    z_xmin_scale=1,           # scale the min
    compute=True,             # enable computation
    plot=False,               # save one frame
    frame_id=0,               # frame to save
    animate=True,             # enable animation
    save_anim=True,           # save animation
    plot_anim=False,          # plot animation
    save_png=False,           # export all png
    load_data=False,          # load data from a file
    save_data=True,           # save data to a file
    use_pickle=True,          # use pickle or joblib
    data_folder='data/simul',          # folder for data files
    output_file='schrodinger_2d.png',  # output file name
    animation_format='mp4',   # animation format (mp4 or gif)
    verbose=True
)

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    pass
