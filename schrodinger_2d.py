#!/usr/bin/env python3
'''
/************************/
/*   schrodinger_2d.py  */
/*     Version 1.0      */
/*      2024/08/12      */
/************************/
'''
import argparse
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import os
import pickle
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import erf
import sys
import time

from mod_config_2d import cfg, p2, p2_changes_load
from mod_config import palette

c = palette
p = p2
p_changes_load = p2_changes_load


def Cap(z, z_min, z_max, width, left=True, right=True):
    z_range = z_max - z_min
    cap_width = width * z_range
    result = 0
    if left:
        left_value = np.maximum(0, (z_min + cap_width - z) / cap_width)
        match p.cap_type:
            case 0:
                result += p.absorbing_strength * (
                    left_value**(int(p.cap_poly)))
            case 1:
                a = p.cap_opt_a
                left_optimal = 0.5 * p.absorbing_strength * (
                    1 + erf(a * left_value - 1))
                result += left_optimal
    if right:
        right_value = np.maximum(0, (z - (z_max - cap_width)) / cap_width)
        match p.cap_type:
            case 0:
                result += p.absorbing_strength * (
                    right_value**(int(p.cap_poly)))
            case 1:
                a = p.cap_opt_a
                right_optimal = 0.5 * p.absorbing_strength * (
                    1 + erf(a * right_value - 1))
                result += right_optimal
    if not left and not right:
        raise ValueError("Either left or right must be True")
    return result


class WavepacketSimulation:
    def __init__(self, outfile, Nx, Ny,
                 x_min, x_max, y_min, y_max, dt, t_max, x0, y0,
                 sigma_x, sigma_y, kx, ky):
        # grid setup
        self.Nx, self.Ny = Nx, Ny
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.x = np.linspace(x_min, x_max, Nx)
        self.y = np.linspace(y_min, y_max, Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        # time parameters
        self.dt = dt
        self.t_max = t_max
        # self.num_frames = int(t_max / dt)
        self.num_frames = int(p.total_duration * p.fps)
        tsteps = int(t_max / dt)
        if tsteps < self.num_frames:
            self.num_frames = tsteps
        self.tsteps_save = int(tsteps / self.num_frames)
        # align the number of steps to the number of frames
        # this reduce the computation time and it is useful
        # to check the simulation quality. In general should
        # set to false
        if cfg.dev_simul:
            print("dev_simul activated. using each computed step")
            self.tsteps_save = 1
        if cfg.verbose:
            print(f"dt: {self.dt}")
            print(f"num steps to compute: {tsteps}")
            print(f"num frames: {self.num_frames}")
            print(f"saving each {self.tsteps_save} steps")
        self.psi_plot = []
        # wavepacket parameters
        self.x0, self.y0 = x0, y0
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.kx, self.ky = kx, ky
        # output
        self._outfile = outfile
        # initialize variables
        self.perc = None
        self.start_time = None
        # initialize simulation
        self.initialize_simulation()

    @property
    def outfile(self):
        return self._outfile

    @outfile.setter
    def outfile(self, value):
        self._outfile = value

    def initialize_simulation(self):
        self.psi = self.psi_0(self.X, self.Y).flatten()
        self.L = self.create_laplacian_matrix()
        self.V_matrix = sparse.diags(self.V(self.X, self.Y).flatten())
        I_M = sparse.eye(self.Nx * self.Ny)
        self.A = I_M + 0.5j * self.dt * (self.L - self.V_matrix)
        self.B = I_M - 0.5j * self.dt * (self.L - self.V_matrix)

    def psi_0(self, x, y):
        return np.exp(-((x - self.x0)**2 / (2 * self.sigma_x**2)
                        + (y - self.y0)**2 / (2 * self.sigma_y**2)
                        )) * np.exp(1j * (self.kx * x + self.ky * y))

    def kinetic_energy(self):
        # reshape self.psi to match the original 2D shape if necessary
        psi_reshaped = self.psi.reshape(self.Ny, self.Nx)
        # compute the Laplacian of the reshaped psi
        laplacian = (np.gradient(np.gradient(psi_reshaped, self.dx, axis=0),
                                 self.dx, axis=0) +
                     np.gradient(np.gradient(psi_reshaped, self.dy, axis=1),
                                 self.dy, axis=1))
        return -0.5 * np.sum(np.conj(psi_reshaped) * laplacian) * \
            self.dx * self.dy

    def potential_energy(self):
        # reshape self.psi to match the original 2D shape if necessary
        psi_reshaped = self.psi.reshape(self.Ny, self.Nx)
        # get the potential from the V function
        V = self.V(self.X, self.Y)
        return np.sum(np.conj(psi_reshaped) * V * psi_reshaped) *\
            self.dx * self.dy

    def total_energy(self):
        # calculate kinetic and potential energies using self.psi
        T = self.kinetic_energy()
        V = self.potential_energy()
        total_energy = T + V
        return total_energy

    def V(self, x, y):
        V_real = np.zeros_like(x)
        if p.middle_barrier and not p.slits:
            # apply the barrier height to the region around
            # the barrier_center within the specified width
            V_real += (np.abs(x - p.barrier_center) <
                       p.barrier_width) * p.barrier_height
        if p.middle_barrier and p.slits:
            # apply several barriers to the region around
            # the barrier_center within the specified width
            # allowing therefore slits
            for start, end in zip(p.barriers_start, p.barriers_end):
                # Ensure correct range regardless of order of start and end
                V_real += ((np.abs(x - p.barrier_center) < p.barrier_width) &
                           (y >= min(start, end)) &
                           (y <= max(start, end))) * p.barrier_height
        if p.infinite_barrier:
            return V_real
        else:
            V_imag = np.zeros_like(x)
            match p.absorbing_method:
                case 0:
                    width_x = p.absorbing_width_x * (
                        self.x_max - self.x_min)
                    width_y = p.absorbing_width_y * (
                        self.y_max - self.y_min)
                    strength = p.absorbing_strength
                    if p.absorbing_xmin:
                        V_imag += strength * (1 - np.tanh((
                            x - self.x_min) / width_x)**2)
                    if p.absorbing_xmax:
                        V_imag += strength * (1 - np.tanh((
                            self.x_max - x) / width_x)**2)
                    if p.absorbing_ymin:
                        V_imag += strength * (1 - np.tanh((
                            y - self.y_min) / width_y)**2)
                    if p.absorbing_ymax:
                        V_imag += strength * (1 - np.tanh((
                            self.y_max - y) / width_y)**2)
                case 1:
                    if p.absorbing_xmin or p.absorbing_xmax:
                        V_imag += Cap(x, self.x_min, self.x_max,
                                      p.absorbing_width_x,
                                      p.absorbing_xmin,
                                      p.absorbing_xmax)
                    if p.absorbing_ymin or p.absorbing_ymax:
                        V_imag += Cap(y, self.y_min, self.y_max,
                                      p.absorbing_width_y,
                                      p.absorbing_ymin,
                                      p.absorbing_ymax)
                case _:
                    raise ValueError("Unsupported smoothing "
                                     f"{p.absorbing_method}")
            return V_real + 1j * V_imag

    def create_laplacian_matrix(self):
        if p.periodic_boundary:
            diags = [-self.Ny, -1, 0, 1, self.Ny]
            data = [
                np.ones(self.Nx * self.Ny) / self.dy**2,
                np.ones(self.Nx * self.Ny) / self.dx**2,
                -2 * np.ones(self.Nx * self.Ny) * (
                    1 / self.dx**2 + 1 / self.dy**2),
                np.ones(self.Nx * self.Ny) / self.dx**2,
                np.ones(self.Nx * self.Ny) / self.dy**2
            ]
            data[1][-1::self.Ny] = data[3][0::self.Ny] = 1 / self.dx**2
            data[0][-self.Ny:] = data[4][:self.Ny] = 1 / self.dy**2
            return sparse.diags(
                data, diags, shape=(self.Nx * self.Ny, self.Nx * self.Ny),
                format='csr')
        else:
            L = sparse.lil_matrix((self.Nx * self.Ny, self.Nx * self.Ny))
            for i in range(self.Ny):
                for j in range(self.Nx):
                    index = i * self.Nx + j
                    L[index, index] = -2 * (1 / self.dx**2 + 1 / self.dy**2)
                    if i > 0:
                        L[index, index - self.Nx] = 1 / self.dy**2
                    if i < self.Ny - 1:
                        L[index, index + self.Nx] = 1 / self.dy**2
                    if j > 0:
                        L[index, index - 1] = 1 / self.dx**2
                    if j < self.Nx - 1:
                        L[index, index + 1] = 1 / self.dx**2
            return L.tocsr()

    def compute(self):
        self.perc = 0
        self.start_time = time.time()
        for i in range(self.num_frames):
            for _ in range(self.tsteps_save):
                # avance the Simulation the necessary number of substeps
                # which will not be saved
                self.psi_copy = self.psi.copy()
                self.psi = spsolve(self.A, self.B @ self.psi_copy)
            # save for the subsequent plot
            self.psi_plot.append(self.psi.copy())
            # self.psi_plot.extend([self.psi])
            if cfg.plot_all_frames:
                energy = self.total_energy()
                print(f"Total Energy of the wave packet: {energy.real:.4}")
                print(f"plotting frame {i}")
                self.plot(i, self._outfile.replace('.png', f'_{i:03d}.png'))
            if cfg.verbose:
                perc = (i + 1) / self.num_frames * 100
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
                    print(f"completed {int(perc)}% of the computation, "
                          f"elapsed {formatted_time} [{current_time}]")

    def __init_plot(self):
        plot_psi = self.psi_plot[0]
        cgray = (0.83, 0.83, 0.83)
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
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # specific visualization option for absorbing boundaries
        if not p.infinite_barrier:
            x_min_limit = p.x_min * (1 - p.absorbing_width_x) \
                if p.absorbing_xmin else p.x_min
            x_max_limit = p.x_max * (1 - p.absorbing_width_x) \
                if p.absorbing_xmax else p.x_max
            y_min_limit = p.y_min * (1 - p.absorbing_width_y) \
                if p.absorbing_ymin else p.y_min
            y_max_limit = p.y_max * (1 - p.absorbing_width_y) \
                if p.absorbing_ymax else p.y_max
            if not cfg.display_all_d:
                # restrict the visualization domain
                ax.set_xlim(x_min_limit, x_max_limit)
                ax.set_ylim(y_min_limit, y_max_limit)
            if cfg.display_all_d and cfg.show_smooth_d:
                # plot the smoothing domain
                rect = patches.Rectangle(
                    (x_min_limit, y_min_limit),
                    x_max_limit - x_min_limit,
                    y_max_limit - y_min_limit,
                    linewidth=2, edgecolor=c.o, facecolor='none')
                ax.add_patch(rect)

        if cfg.add_screen:
            # display a screen location that could be used for data
            # collection (double slit for example)
            polygon = patches.Polygon(
                cfg.screen_data, closed=True, fill=True, edgecolor=cgray,
                facecolor=c.b, alpha=0.4, linewidth=0.7)
            ax.add_patch(polygon)

        if p.middle_barrier and not p.slits:
            # create the rectangle representing the barrier
            barrier_rect = patches.Rectangle(
                (p.barrier_center - p.barrier_width, p.y_min),
                2 * p.barrier_width, p.y_max - p.y_min,
                linewidth=2, edgecolor=cgray, facecolor='none')
            ax.add_patch(barrier_rect)

        if p.middle_barrier and p.slits:
            # create multiple vertical barriers with gaps (slits) between them
            for start, end in zip(p.barriers_start, p.barriers_end):
                barrier_rect = patches.Rectangle(
                    (p.barrier_center - p.barrier_width, start),
                    2 * p.barrier_width, end - start,
                    linewidth=2, edgecolor=cgray, facecolor='none')
                ax.add_patch(barrier_rect)

        if p.infinite_barrier:
            rect = patches.Rectangle(
                (p.x_min, p.y_min), p.x_max - p.x_min, p.y_max - p.y_min,
                linewidth=3, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
        if cfg.plot_prob:
            if cfg.fix_min_max:
                # compute minimum and maximum over the entire data
                vmax_value = cfg.z_xmax_scale * np.max(np.abs(plot_psi**2))
                vmin_value = cfg.z_xmin_scale * np.min(np.abs(plot_psi**2))
                # plot the probability distribution
                self.img = ax.imshow(
                    np.abs(plot_psi.reshape(self.Ny, self.Nx))**2,
                    extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                    vmin=vmin_value, vmax=vmax_value, cmap='hot')
            else:
                self.img = ax.imshow(
                    np.abs(plot_psi.reshape(self.Ny, self.Nx))**2,
                    extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                    cmap='hot')
        else:
            # set the figure background color
            self.fig.patch.set_facecolor('black')
            # set the axes background color
            ax.set_facecolor('black')
            # plot the modulus of the wavefunction and the phase
            psi = plot_psi.reshape(self.Ny, self.Nx)
            magnitude = np.abs(psi)
            phase = np.angle(psi)
            normalized_phase = (phase + np.pi) / (2 * np.pi)
            hsv_image = cm.hsv(normalized_phase)
            hsv_image[..., 3] = np.clip(magnitude / np.nanmax(magnitude), 0, 1)
            if cfg.fix_min_max:
                # compute minimum and maximum over the entire data
                vmax_value = cfg.z_xmax_scale * np.max(np.abs(plot_psi))
                vmin_value = cfg.z_xmin_scale * np.min(np.abs(plot_psi))
                self.img = ax.imshow(
                    hsv_image, origin='lower',
                    extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                    vmin=vmin_value, vmax=vmax_value)
            else:
                self.img = ax.imshow(
                    hsv_image, origin='lower',
                    extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        ax.set_aspect('auto')
        ax.axis('off')

    def __animate_frame(self, frame, is_animation=True, is_pngexport=False):
        if cfg.plot_prob:
            self.img.set_array(np.abs(
                self.psi_plot[frame].reshape(self.Ny, self.Nx))**2)
        else:
            psi = self.psi_plot[frame].reshape(self.Ny, self.Nx)
            magnitude = np.abs(psi)
            phase = np.angle(psi)
            normalized_phase = (phase + np.pi) / (2 * np.pi)
            hsv_image = cm.hsv(normalized_phase)
            hsv_image[..., 3] = np.clip(
                magnitude / np.nanmax(magnitude), 0, 1)
            self.img.set_array(hsv_image)
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

        return self.img,

    def plot(self, nframe=cfg.frame_id, fname=None):
        if fname is None:
            fname = self._outfile
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
            outfile_a = f"{base}.{animation_format}"
            if animation_format == 'mp4':
                anim.save(outfile_a, writer='ffmpeg')
            elif animation_format == 'gif':
                anim.save(outfile_a, writer='imagemagick')
        if cfg.plot_anim:
            plt.show()

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


def ConsistencyChecks():
    if len(p.barriers_start) != len(p.barriers_end):
        raise ValueError(
            "barriers_start and barriers_end must ""have the same length")
    if not p.middle_barrier and p.slits:
        raise ValueError("Slits cannot be defined without a middle barrier")
    if p.absorbing_method == 1:
        match p.cap_type:
            case 0 | 1:
                pass  # valid cap_type
            case _:
                raise ValueError(
                    "cap_type must be 0 (polynomial)"
                    " or 1 (optimal) for absorbing_method 1")


def make_plot(outfile: str):
    global p
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} " \
        r"\usepackage{amsmath} \usepackage{helvet}"
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica",
        "font.sans-serif": "Helvetica"
    })
    plt.rcParams['animation.convert_path'] = 'magick'
    if cfg.load_data:
        folder = cfg.data_folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        simul_dir = os.path.join(script_dir, folder)
        with open(simul_dir + '/config.pkl', 'rb') as file:
            p = pickle.load(file)
        # update any value in the config if needed
        for key, value in p_changes_load.__dict__.items():
            setattr(p, key, value)
        if cfg.verbose:
            print(f"Loading data ({simul_dir}/data.pkl)")
        with open(simul_dir + '/data.pkl', 'rb') as file:
            sim = pickle.load(file)
            # reset the output file
            sim.outfile = outfile
        if cfg.verbose:
            # check the input data
            ConsistencyChecks()
            energy = sim.total_energy()
            print(f"Total Energy of the wave packet: {energy.real:.4}")
        if cfg.verbose and p.middle_barrier:
            print(f"Barrier height: {p.barrier_height:.4}")
    else:
        # Do not compute or serialize if load
        if cfg.verbose:
            # check the input data
            ConsistencyChecks()
        if cfg.high_res_dt:
            dt = p.dt_hr
        else:
            dt = p.dt_lr
        if cfg.high_res_grid:
            Nx = p.Nx_hr
            Ny = p.Ny_hr
        else:
            Nx = p.Nx_lr
            Ny = p.Ny_lr
        sim = WavepacketSimulation(
            outfile, Nx, Ny, p.x_min, p.x_max, p.y_min, p.y_max,
            dt, p.t_max, p.x0, p.y0, p.sigma_x, p.sigma_y, p.kx, p.ky)
        if cfg.verbose:
            energy = sim.total_energy()
            print(f"Total Energy of the wave packet: {energy.real:.4}")
        if cfg.verbose and p.middle_barrier:
            print(f"Barrier height: {p.barrier_height:.4}")
        sim.compute()
        if cfg.save_data:
            folder = cfg.data_folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            simul_dir = os.path.join(script_dir, folder)
            if cfg.verbose:
                print(f"Saving config and data ({simul_dir})")
            if not os.path.exists(simul_dir):
                os.makedirs(simul_dir)
            with open(simul_dir + '/config.pkl', 'wb') as file:
                pickle.dump(p, file)
            with open(simul_dir + '/data.pkl', 'wb') as file:
                pickle.dump(sim, file)
    if cfg.animate:
        sim.animate()
    if cfg.save_png:
        sim.export_png()
    if cfg.plot:
        sim.plot()


def main():
    parser = argparse.ArgumentParser(
        description='schrodinger 2d simulation')
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
