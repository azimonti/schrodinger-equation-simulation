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
import sys

from mod_config_2d import cfg, p2

p = p2


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
        self.num_frames = p.total_duration * p.fps
        tsteps = int(t_max / dt)
        if tsteps < self.num_frames:
            self.num_frames = tsteps
        self.tsteps_save = int(tsteps / self.num_frames)
        # align the number of steps to the number of frames
        # this reduce the computation time and it is useful
        # to check the simulation quality. In general should
        # set to false
        if cfg.dev_simul:
            self.tsteps_save = 1
        if cfg.verbose:
            print(f"saving each {self.tsteps_save} steps")
        self.psi_plot = []

        # Wavepacket parameters
        self.x0, self.y0 = x0, y0
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.kx, self.ky = kx, ky

        # output
        self.outfile = outfile

        # initialize variables
        self.perc = None

        # initialize simulation
        self.initialize_simulation()

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
        # Reshape self.psi to match the original 2D shape if necessary
        psi_reshaped = self.psi.reshape(self.Ny, self.Nx)
        # Compute the Laplacian of the reshaped psi
        laplacian = (np.gradient(np.gradient(psi_reshaped, self.dx, axis=0),
                                 self.dx, axis=0) +
                     np.gradient(np.gradient(psi_reshaped, self.dy, axis=1),
                                 self.dy, axis=1))

        return -0.5 * np.sum(np.conj(psi_reshaped) * laplacian) * \
            self.dx * self.dy

    def potential_energy(self):
        # Reshape self.psi to match the original 2D shape if necessary
        psi_reshaped = self.psi.reshape(self.Ny, self.Nx)

        # Get the potential from the V function
        V = self.V(self.X, self.Y)

        return np.sum(np.conj(psi_reshaped) * V * psi_reshaped) *\
            self.dx * self.dy

    def total_energy(self):
        # Calculate kinetic and potential energies using self.psi
        T = self.kinetic_energy()
        V = self.potential_energy()

        total_energy = T + V
        return total_energy

    def V(self, x, y):
        V_real = np.zeros_like(x)
        # set a finite barrier in the middle of the x
        # Set a finite barrier in the middle of the x direction
        if cfg.middle_barrier:
            # Apply the barrier height to the region around
            # the center within the specified width
            V_real += (np.abs(x - p.barrier_center) <
                       p.barrier_width) * p.barrier_height
        if cfg.infinite_barrier:
            return V_real
        else:
            width_x = 0.1 * (self.x_max - self.x_min)
            width_y = 0.1 * (self.y_max - self.y_min)
            strength = 5.0
            V_imag = np.zeros_like(x)
            V_imag += strength * (1 - np.tanh((x - self.x_min) / width_x)**2)
            V_imag += strength * (1 - np.tanh((self.x_max - x) / width_x)**2)
            V_imag += strength * (1 - np.tanh((y - self.y_min) / width_y)**2)
            V_imag += strength * (1 - np.tanh((self.y_max - y) / width_y)**2)
            return V_real + 1j * V_imag

    def create_laplacian_matrix(self):
        if cfg.periodic_boundary:
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
        for i in range(self.num_frames):
            for _ in range(self.tsteps_save):
                # avance the Simulation the necessary number of substeps
                # which will not be saved
                self.psi = spsolve(self.A, self.B @ self.psi)
            # save for the subsequent plot
            self.psi_plot.extend([self.psi])
            if cfg.verbose:
                perc = (i + 1) / self.num_frames * 100
                if perc // 10 > self.perc // 10:
                    self.perc = perc
                    print(f"completed {int(perc)}% of the computation")

    def animate(self):
        self.perc = 0
        if cfg.high_res_plot:
            fig, ax = plt.subplots(dpi=300)
        else:
            fig, ax = plt.subplots()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if cfg.middle_barrier:
            color = (0.83, 0.83, 0.83)
            # Create the rectangle representing the barrier
            barrier_rect = patches.Rectangle(
                (p.barrier_center - p.barrier_width, p.y_min),
                2 * p.barrier_width, p.y_max - p.y_min,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(barrier_rect)
        if cfg.infinite_barrier:
            rect = patches.Rectangle(
                (p.x_min, p.y_min), p.x_max - p.x_min, p.y_max - p.y_min,
                linewidth=3, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
        if cfg.plot_prob:
            # plot the probability distribution
            img = ax.imshow(
                np.abs(self.psi.reshape(self.Ny, self.Nx))**2,
                extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                cmap='hot')
        else:
            # Set the figure background color
            fig.patch.set_facecolor('black')
            # Set the axes background color
            ax.set_facecolor('black')
            # plot the modulus of the wavefunction and the phase
            psi = self.psi.reshape(self.Ny, self.Nx)
            magnitude = np.abs(psi)
            phase = np.angle(psi)
            normalized_phase = (phase + np.pi) / (2 * np.pi)
            hsv_image = cm.hsv(normalized_phase)
            hsv_image[..., 3] = np.clip(magnitude / np.nanmax(magnitude), 0, 1)
            img = ax.imshow(
                hsv_image, origin='lower',
                extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        ax.axis('off')

        def animate_frame(frame):
            if cfg.plot_prob:
                img.set_array(np.abs(
                    self.psi_plot[frame].reshape(self.Ny, self.Nx))**2)
            else:
                psi = self.psi_plot[frame].reshape(self.Ny, self.Nx)
                magnitude = np.abs(psi)
                phase = np.angle(psi)
                normalized_phase = (phase + np.pi) / (2 * np.pi)
                hsv_image = cm.hsv(normalized_phase)
                hsv_image[..., 3] = np.clip(
                    magnitude / np.nanmax(magnitude), 0, 1)
                img.set_array(hsv_image)
            if cfg.verbose:
                perc = (frame + 1) / self.num_frames * 100
                if perc // 10 > self.perc // 10:
                    self.perc = perc
                    print(f"completed {int(perc)}% of the animation")
            return img,

        anim = FuncAnimation(
            fig, animate_frame, frames=self.num_frames,
            interval=1000 / p.fps, blit=True)
        if cfg.save_anim:
            base, ext = self.outfile.rsplit('.', 1)
            animation_format = cfg.animation_format
            outfile_a = f"{base}.{animation_format}"
            if animation_format == 'mp4':
                anim.save(outfile_a, writer='ffmpeg')
            elif animation_format == 'gif':
                anim.save(outfile_a, writer='imagemagick')
        if cfg.plot:
            plt.show()


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
        if cfg.verbose:
            print(f"Loading params ({simul_dir}/config.pkl)")
        if not os.path.exists(simul_dir):
            raise FileNotFoundError(f"path not found: {simul_dir}")
        with open(simul_dir + '/config.pkl', 'rb') as file:
            p = pickle.load(file)
        if cfg.verbose:
            print(f"Loading data ({simul_dir}/data.pkl)")
        with open(simul_dir + '/data.pkl', 'rb') as file:
            sim = pickle.load(file)
    else:
        # Do not compute or serialize if load
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
        if cfg.verbose and cfg.middle_barrier:
            energy = sim.total_energy()
            print(f"Total Energy of the wave packet: {energy.real:.4}")
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
        ofile = tmp_dir + "/schrodinger_2d.png"
    make_plot(ofile)


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
