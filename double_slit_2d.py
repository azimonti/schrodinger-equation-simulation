#!/usr/bin/env python3
'''
/************************/
/*   double_slit_2d.py  */
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

from mod_config_double_slit_2d import cfg, p2_changes_load_ds2d
from mod_config import palette
from schrodinger_2d import WavepacketSimulation

if cfg.use_pickle:
    from pickle import load, dump
    ext = 'pkl'
else:
    from joblib import load, dump
    ext = 'joblib'


c = palette
p_changes_load = p2_changes_load_ds2d


class DoubleSlitSimulation:
    def __init__(self, outfile, wavepacket: WavepacketSimulation):
        self.wp = wavepacket
        # output
        self._outfile = outfile
        # initialize variables
        self.perc = None
        self.start_time = None
        self.x_min = wavepacket.x_min
        self.x_max = wavepacket.x_max
        self.y_min = wavepacket.y_min
        self.y_max = wavepacket.y_max
        self.dx = wavepacket.dx
        self.dy = wavepacket.dy
        self.num_frames = int(p.total_duration * p.fps)
        self.Nx = wavepacket.Nx
        self.Ny = wavepacket.Ny

    @property
    def outfile(self):
        return self._outfile

    @outfile.setter
    def outfile(self, value):
        self._outfile = value

    def introspection(self):
        print(f"Number of points: {len(self.screen_data_total)}")

    def line_cells_crossed(self):
        self.crossed_nx = []
        self.crossed_ny = []
        # Unpack capture data from the global cfg
        (x1, y1), (x2, y2) = cfg.capture_data

        # Calculate the grid coordinates of the endpoints
        x1_idx = int((x1 - self.x_min) // self.dx)
        y1_idx = int((y1 - self.y_min) // self.dy)
        x2_idx = int((x2 - self.x_min) // self.dx)
        y2_idx = int((y2 - self.y_min) // self.dy)

        # Bresenham's line algorithm adapted to this grid
        cells_crossed = []
        seen_cells = set()

        dx = abs(x2_idx - x1_idx)
        dy = abs(y2_idx - y1_idx)
        sx = 1 if x1_idx < x2_idx else -1
        sy = 1 if y1_idx < y2_idx else -1
        err = dx - dy

        x, y = x1_idx, y1_idx
        while True:
            # Store the actual grid indices
            self.crossed_nx.append(x)
            self.crossed_ny.append(y)
            # compute cell mid point
            x_mid = self.x_min + x * self.dx + 0.5 * self.dx
            y_mid = self.y_min + y * self.dy + 0.5 * self.dy
            cell = (x_mid, y_mid)
            if cell in seen_cells:
                raise ValueError(f"Duplicate cell detected at {cell}")
            seen_cells.add(cell)
            cells_crossed.append(cell)
            if x == x2_idx and y == y2_idx:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return len(cells_crossed), cells_crossed

    def compute(self):
        # Initialize the screen_data array
        self.screen_data_total = np.zeros(len(self.crossed_nx))
        self.screen_data_plot = []

        # Loop over each snapshot in psi_plot
        for psi in self.wp.psi_plot:
            # Reshape the 1D wavefunction array to 2D
            psi = psi.reshape(self.Ny, self.Nx)
            # initialize a temporary array to accumulate data for this snapshot
            temp_data = np.zeros(len(self.crossed_nx))
            # loop over each cell in the crossed path
            for i, (nx, ny) in enumerate(zip(self.crossed_nx,
                                             self.crossed_ny)):
                if cfg.plot_prob:
                    # calculate the probability density
                    data = np.abs(psi[ny, nx])**2
                else:
                    # calculate the modulus of the wavefunction
                    data = np.abs(psi[ny, nx])
                # accumulate data for this snapshot
                temp_data[i] = data
            # Append the snapshot's data to the plot list as a 1D array
            self.screen_data_plot.append(temp_data.copy())
            # Accumulate this snapshot's data into the total
            self.screen_data_total += temp_data

    def __init_plot(self):
        dpi = 300 if cfg.high_res_plot else 100
        if cfg.fig_4k:
            figsize = (3840 / dpi, 2160 / dpi)
        else:
            figsize = (1920 / dpi, 1080 / dpi)
        self.fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        # the screen is assumed vertical in the y direction
        ax.set_xlim(self.y_min, self.y_max)
        ax.set_ylim(0, max(self.screen_data_total) * 1.01)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # init the total data
        self.screen_data_total_tmp = np.zeros(len(self.crossed_nx))
        # Calculate real distances along the y-axis
        real_distances = self.y_min + np.array(self.crossed_ny) * self.dy
        # Sort by real distances
        self.sorted_indices = np.argsort(real_distances)
        self.sorted_distances = real_distances[self.sorted_indices]
        y0 = np.zeros(len(self.sorted_distances))
        self.curve1 = ax.plot(self.sorted_distances, y0, color=c.b,
                              linestyle="-", linewidth=3)[0]
        if cfg.plot_secondary:
            self.curve2 = ax.plot(self.sorted_distances, y0, color=c.o,
                                  linestyle="-", linewidth=3)[0]
        plt.tight_layout()

    def __init_plot2(self):
        dpi = 300 if cfg.high_res_plot else 100
        if cfg.fig_4k:
            figsize = (3840 / dpi, 2160 / dpi)
        else:
            figsize = (1920 / dpi, 1080 / dpi)
        self.fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        # Convert grid indices to real x and y coordinates
        self.crossed_x = [self.x_min + nx * self.dx + 0.5 * self.dx
                          for nx in self.crossed_nx]
        self.crossed_y = [self.y_min + ny * self.dy + 0.5 * self.dy
                          for ny in self.crossed_ny]
        self.curve3 = ax.scatter(self.crossed_x, self.crossed_y,
                                 c=np.zeros_like(self.crossed_x), cmap='hot',
                                 vmin=0, vmax=max(self.screen_data_total))
        # the screen is assumed vertical in the y direction
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.tight_layout()

    def __animate_frame(self, frame, is_animation=True, is_pngexport=False):
        sorted_data = self.screen_data_plot[frame][self.sorted_indices]
        self.screen_data_total_tmp += sorted_data
        self.curve1.set_ydata(self.screen_data_total_tmp)
        if cfg.plot_secondary:
            self.curve2.set_ydata(sorted_data)
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
        if cfg.plot_secondary:
            return self.curve1, self.curve2
        else:
            return (self.curve1,)

    def plot(self, nframe=cfg.frame_id):
        self.perc = 0
        self.start_time = time.time()
        self.__init_plot()
        for n in range(nframe):
            self.__animate_frame(n, False, True)
        plt.savefig(self._outfile.replace('.png', f'_{nframe}.png'), dpi=300)
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
            outfile_a = f"{base}.{animation_format}"
            if animation_format == 'mp4':
                anim.save(outfile_a, writer='ffmpeg')
            elif animation_format == 'gif':
                anim.save(outfile_a, writer='imagemagick')
        if cfg.plot_anim:
            plt.show()
        plt.close()

        self.__init_plot()
        sorted_data = self.screen_data_total[self.sorted_indices]
        self.curve1.set_ydata(sorted_data)
        plt.savefig(self._outfile, dpi=300)
        plt.close()

        self.__init_plot2()
        self.curve3.set_array(self.screen_data_total)
        plt.savefig(self._outfile.replace('.png', '_2d.png'), dpi=300)
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
            self.__init_plot()
            self.__animate_frame(nframe, False, True)
            fname = os.path.join(tmp_dir, f"{base_name}_{nframe:05d}.png")
            plt.savefig(fname, dpi=300)
            plt.close()


def make_plot(outfile: str):
    global p
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
    with open(f'{simul_dir}/config_s2d.{ext}', 'rb') as file:
        p = load(file)
    # update any value in the config if needed
    for key, value in p_changes_load.__dict__.items():
        setattr(p, key, value)
    if cfg.verbose:
        print(f"Loading data ({simul_dir}/data_s2d.{ext})")
    with open(f'{simul_dir}/data_s2d.{ext}', 'rb') as file:
        wavepacket = load(file)
    sim = DoubleSlitSimulation(outfile, wavepacket)
    num_cells, midpoints = sim.line_cells_crossed()
    if cfg.verbose:
        wavepacket.introspection()
        print("Number of cells crossed:", num_cells)
        if cfg.print_crossed_cells:
            for i in range(0, len(midpoints), max(1, len(midpoints) // 10)):
                print(f"Midpoint {i}: {midpoints[i]}")
            print("Crossed nx:", sim.crossed_nx)
            print("Crossed ny:", sim.crossed_ny)
    sim.compute()
    if cfg.animate:
        sim.animate()
    if cfg.save_png:
        sim.export_png()
    if cfg.plot:
        sim.plot()
    # remove data no longer needed before saving
    del sim.wp
    if cfg.save_data:
        folder = cfg.data_folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        simul_dir = os.path.join(script_dir, folder)
        if cfg.verbose:
            print(f"Saving config and data ({simul_dir})")
        if not os.path.exists(simul_dir):
            os.makedirs(simul_dir)
        with open(f'{simul_dir}/config_ds2d.{ext}', 'wb') as file:
            dump(p, file)
        with open(f'{simul_dir}/data_ds2d.{ext}', 'wb') as file:
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
