#!/usr/bin/env python3
'''
/********************************/
/* mod_config_double_slit_2d.py */
/*         Version 1.0          */
/*          2024/08/31          */
/********************************/
'''
import sys
from types import SimpleNamespace

p2_changes_load_ds2d = SimpleNamespace()

p2_add_load_eb = SimpleNamespace(
    total_duration=90,          # total duration of the animation
    fps=60                      # frames per second of the animation
)

p2_changes_load_eb = SimpleNamespace(
    electrons_nb=1.5e4          # number of electrons in the beam
)

cfg = SimpleNamespace(
    plot_prob=True,             # plot probability or wavefunction
    plot_secondary=False,       # plot each step values
    high_res_plot=True,         # enable high resolution simulation plot
    fig_4k=False,               # use 4k resolution
    capture_data=[[10, -14.95], [10, 14.95]],   # capture location
    print_crossed_cells=False,  # print details of the cells which are crossed
    plot=False,                 # save one frame
    frame_id=0,                 # frame to save
    animate=True,               # enable animation
    save_anim=True,             # save animation
    plot_anim=False,            # plot animation
    save_png=False,             # export all png
    save_beam_count=True,       # export the beam distribution
    save_data=True,             # save data to a file
    use_pickle=True,            # use pickle or joblib
    data_folder='data/simul',               # folder for data files
    output_file='schrodinger_beam_2d.png',  # output file name
    animation_format='mp4',     # animation format (mp4 or gif)
    verbose=True
)

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    pass
