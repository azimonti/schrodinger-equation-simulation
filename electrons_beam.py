#!/usr/bin/env python3
'''
/************************/
/*   electrons_beam.py  */
/*     Version 1.0      */
/*      2024/08/31      */
/************************/
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import time

from mod_config_double_slit_2d import cfg, p2_changes_load_eb, p2_add_load_eb
from mod_config import palette
from double_slit_2d import DoubleSlitSimulation

if cfg.use_pickle:
    from pickle import load, dump
    ext = 'pkl'
else:
    from joblib import load, dump
    ext = 'joblib'


c = palette
p_changes_load = p2_changes_load_eb
p_add_load = p2_add_load_eb


class ElectronBeamSimulation:
    def __init__(self, outfile, doubleslit: DoubleSlitSimulation):
        # output
        self._outfile = outfile
        # normalize perc to make the sum equal to 1
        self.probability = [p / sum(doubleslit.screen_data_total)
                            for p in doubleslit.screen_data_total]
        self.x_min = doubleslit.x_min
        self.x_max = doubleslit.x_max
        self.y_min = doubleslit.y_min
        self.y_max = doubleslit.y_max
        self.dx = doubleslit.dx
        self.dy = doubleslit.dy
        self.num_frames = int(p.total_duration * p.fps)
        self.Nx = doubleslit.Nx
        self.Ny = doubleslit.Ny
        self.crossed_ny = doubleslit.crossed_ny
        # initialize variables
        self.perc = None
        self.start_time = None
        if cfg.verbose:
            print(f"Sum of probability: {sum(self.probability):.3f}")

    @property
    def outfile(self):
        return self._outfile

    @outfile.setter
    def outfile(self, value):
        self._outfile = value

    def compute(self):
        # Initialize the screen_data array
        self.electrons = []
        self.electrons_height = []
        self.electrons_count = np.zeros(len(self.probability))
        for n in range(int(p.electrons_nb)):
            # Randomly choose a bucket based on the probability distribution
            bucket = np.random.choice(len(self.probability),
                                      p=self.probability)
            # Update the electrons count
            self.electrons_count[bucket] += 1
            # Keep track of the electron bucket assignment
            self.electrons.append(bucket)
            # Compute a random height on the electron
            height = np.random.uniform(0, 1)
            self.electrons_height.append(height)

    def __init_plot(self):
        if cfg.fig_4k:
            if cfg.high_res_plot:
                self.fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=300)
            else:
                self.fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=300)
        else:
            if cfg.high_res_plot:
                self.fig, ax = plt.subplots(dpi=300)
            else:
                self.fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        # the screen is assumed vertical in the y direction
        ax.set_xlim(self.y_min, self.y_max)
        ax.set_ylim(0, 1.01)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # init the total data
        # Calculate real distances along the y-axis
        real_distances = self.y_min + np.array(self.crossed_ny) * self.dy
        # Sort by real distances
        self.sorted_indices = np.argsort(real_distances)
        self.sorted_distances = real_distances[self.sorted_indices]
        self.curve1 = ax.scatter([], [], color=c.b, s=8,
                                 edgecolor=c.k, linewidth=0.2)
        plt.tight_layout()

    def __init_plot2(self):
        if cfg.fig_4k:
            if cfg.high_res_plot:
                self.fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=300)
            else:
                self.fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=300)
        else:
            if cfg.high_res_plot:
                self.fig, ax = plt.subplots(dpi=300)
            else:
                self.fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        # the screen is assumed vertical in the y direction
        ax.set_xlim(self.y_min, self.y_max)
        ax.set_ylim(0, max(self.electrons_count) * 1.01)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        real_distances = self.y_min + np.array(self.crossed_ny) * self.dy
        # Sort by real distances
        self.sorted_indices = np.argsort(real_distances)
        self.sorted_distances = real_distances[self.sorted_indices]
        y0 = np.zeros(len(self.sorted_distances))
        self.curve2 = ax.plot(self.sorted_distances, y0, color=c.b,
                              linestyle="-", linewidth=3)[0]
        plt.tight_layout()

    def __animate_frame(self, frame, is_animation=True, is_pngexport=False):
        # total number of electrons
        total_electrons = len(self.electrons)

        # Determine how many electrons should be displayed by this frame
        # We calculate this as a proportion of the total number of frames
        electrons_to_show = (frame + 1) * total_electrons // self.num_frames

        # ensure we don't exceed the total number of electrons
        electrons_to_show = min(electrons_to_show, total_electrons)

        # select the electrons that should be shown in this frame
        selected_electrons = self.electrons[:electrons_to_show]

        # use selected_electrons to get the corresponding X and Y data
        x_data = [self.sorted_distances[i] for i in selected_electrons]
        # get the corresponding Y data (heights)
        y_data = self.electrons_height[:electrons_to_show]

        # update scatter plot data
        self.curve1.set_offsets(np.c_[x_data, y_data])

        if cfg.verbose and (is_animation or is_pngexport):
            if is_animation:
                ptext = "the animation"
            else:
                ptext = "png export"
            perc = (frame + 1) / self.num_frames * 100
            if perc // 10 > self.perc // 10:
                self.perc = perc
                elapsed_time = time.time() - self.start_time
                current_time = time.strftime("%H:%M:%S", time.localtime())
                if elapsed_time >= 3600:
                    formatted_time = time.strftime(
                        "%H:%M:%S", time.gmtime(elapsed_time))
                else:
                    formatted_time = time.strftime(
                        "%M:%S", time.gmtime(elapsed_time))
                print(f"completed {int(perc)}% of {ptext}, "
                      f"elapsed {formatted_time} [{current_time}]")
        return (self.curve1,)

    def plot(self, nframe=cfg.frame_id, fname=None):
        if fname is None:
            fname = self._outfile.replace('.png', f'_{nframe}.png')
        self.perc = 0
        self.start_time = time.time()
        self.__init_plot()
        self.__animate_frame(nframe, False, True)
        plt.savefig(fname, dpi=300)
        plt.close()

    def animate(self):
        self.perc = 0
        self.start_time = time.time()
        self.__init_plot()
        anim = FuncAnimation(
            self.fig, self.__animate_frame, frames=self.num_frames,
            interval=1000 / p.fps, blit=True)
        if cfg.save_anim:
            base, ext = self._outfile.rsplit('.', 1)
            animation_format = cfg.animation_format
            outfile_a = f"{base}_beam.{animation_format}"
            if animation_format == 'mp4':
                anim.save(outfile_a, writer='ffmpeg')
            elif animation_format == 'gif':
                anim.save(outfile_a, writer='imagemagick')
        if cfg.plot_anim:
            plt.show()
        plt.close()

        # export the last frame
        self.plot(self.num_frames, self._outfile.replace('.png', '_beam.png'))

    def plot_beam_count(self):
        self.__init_plot2()
        self.curve2.set_ydata(self.electrons_count)
        plt.savefig(self._outfile.replace('.png', '_beam_count.png'), dpi=300)
        plt.close()

    def export_png(self):
        self.perc = 0
        self.start_time = time.time()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, 'tmp', cfg.data_folder)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        base_name = os.path.basename(self._outfile)
        for nframe in range(self.num_frames):
            fname = os.path.join(tmp_dir, f"{base_name}_{nframe:05d}.png")
            self.plot(nframe, fname)


def make_plot(outfile: str):
    global p
    np.random.seed(63746)
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} " \
        r"\usepackage{amsmath} \usepackage{helvet}"
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica"
    })
    plt.style.use('dark_background')
    plt.rcParams['animation.convert_path'] = 'magick'
    folder = cfg.data_folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    simul_dir = os.path.join(script_dir, folder)
    if not os.path.exists(simul_dir):
        raise FileNotFoundError(f"Directory not found: {simul_dir}")
    with open(f'{simul_dir}/config_ds2d.{ext}', 'rb') as file:
        p = load(file)
    # update any value in the config if needed
    for key, value in p_changes_load.__dict__.items():
        setattr(p, key, value)
    # add additional config
    for key, value in p_add_load.__dict__.items():
        setattr(p, key, value)
    if cfg.verbose:
        print(f"Loading data ({simul_dir}/data_s2d.{ext})")
    with open(f'{simul_dir}/data_ds2d.{ext}', 'rb') as file:
        doubleslit = load(file)
    sim = ElectronBeamSimulation(outfile, doubleslit)
    if cfg.verbose:
        doubleslit.introspection()
    sim.compute()
    if cfg.animate:
        sim.animate()
    if cfg.save_png:
        sim.export_png()
    if cfg.plot:
        sim.plot()
    if cfg.save_beam_count:
        sim.plot_beam_count()
    if cfg.save_data:
        folder = cfg.data_folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        simul_dir = os.path.join(script_dir, folder)
        if cfg.verbose:
            print(f"Saving config and data ({simul_dir})")
        if not os.path.exists(simul_dir):
            os.makedirs(simul_dir)
        with open(f'{simul_dir}/config_eb.{ext}', 'wb') as file:
            dump(p, file)
        with open(f'{simul_dir}/data_eb.{ext}', 'wb') as file:
            dump(sim, file)


def main():
    parser = argparse.ArgumentParser(
        description='double slit 2d simulation')
    parser.add_argument('-o', '--ofile', help='output file')
    args = parser.parse_args()
    if args.ofile:
        ofile = args.ofile
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tmp_dir = os.path.join(script_dir, 'tmp')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        ofile = tmp_dir + "/" + cfg.output_file
    make_plot(ofile)


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
