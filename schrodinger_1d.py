#!/usr/bin/env python3
'''
/**********************/
/*  schrodinger_1d.py */
/*    Version 1.1     */
/*    2024/06/07      */
/**********************/
'''
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy import integrate, sparse
from scipy.special import hermite
import sys

from mod_config import cfg, palette, p1, electron_params
from mod_plotter import BasePlotter


c = palette
# select the set of parameters to use
p = electron_params if cfg.small_scale else p1


def create_wavepacket(x):
    # Gaussian shape
    gaussian = np.exp(-(x - p.x0) ** 2 / (2 * p.sigma ** 2))
    # Adding the wave component (momentum)
    psi0 = gaussian * np.exp(-1j * p.p * x / p.hbar)
    # Normalize
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * p.dx)
    return psi0


def create_superposition(x):
    match p.potential:
        case 1:
            # harmonic oscillator
            # characteristic length scale
            a = np.sqrt(p.hbar / (p.m * p.omega))
            psi = sum(np.exp(-1j * n * p.omega * p.p)
                      * (1.0 / np.sqrt(2.0**n * np.math.factorial(n)))
                      * (1.0 / np.pi**0.25)
                      * hermite(n)(x / a)
                      * np.exp(-x**2 / (2 * a**2))
                      for n in cfg.eigenfunctions_list)
            # normalization
            norm = np.sqrt(integrate.simps(np.abs(psi)**2, x))
            psi /= norm
        case 2:
            # infinite high barrier
            L = 2 * p.Vx_bar
            p0 = p.p
            # superposition of eigenfunctions
            psi = sum(np.exp(-1j * n * np.pi * p0 / L)
                      * np.sqrt(2 / L) * np.sin(n * np.pi * (x + p.Vx_bar) / L)
                      for n in cfg.eigenfunctions_list)
            # zero out the wave function outside the well [-p.Vx_bar, p.Vx_bar]
            psi = np.where((x >= -p.Vx_bar) & (x <= p.Vx_bar), psi, 0)
            # normalization
            norm = np.sqrt(np.sum(np.abs(psi)**2) * (x[1] - x[0]))
            psi /= norm
        case _:
            raise NotImplementedError(
                f"Superposition for potential {p.potential} not implemented")
    return psi


def T_wavepacket(psi):
    d2psi_dx2 = np.gradient(np.gradient(psi, p.dx), p.dx)
    T_density = -0.5 * p.hbar ** 2 / p.m * np.conj(psi) * d2psi_dx2
    T = np.sum(T_density.real) * p.dx
    return T


def create_potential(x):
    match p.potential:
        case 0:
            # free space
            return np.zeros(len(x))
        case 1:
            # harmonic oscillator
            return 0.5 * p.m * p.omega**2 * x ** 2
        case 2:
            # infinite high barrier
            V = np.zeros(len(x))
            # with a value too big RK solver is not converging
            # we set the barrier ~1e4 times the value of an average V_0
            V_inf = 2e-14 if cfg.small_scale else 1e4
            V[x <= -p.Vx_bar] = V_inf
            V[x >= p.Vx_bar] = V_inf
            return V
        case 3:
            # infinite right barrier
            V = np.zeros(len(x))
            V[x >= p.Vx_bar] = p.V_barrier
            return V
        case 4:
            # finite right barrier
            V = np.zeros(len(x))
            V[(x >= p.Vx_bar) & (x <= p.Vx_finite_bar)] = p.V_barrier
            return V
        case _:
            raise NotImplementedError(
                f"Potential {p.potential} not implemented")


# Define the time evolution function for solve_ivp
def schrodinger_rhs(t, psi, H):
    dpsi_dt = 1j / p.hbar * (H @ psi)
    return dpsi_dt


class MyPlotter(BasePlotter):
    def __init__(self, params: dict, outfile: str):
        super().__init__(params, outfile)

    def create_axes(self, is_plot: bool):
        ax = self._axs_p if is_plot else self._axs_a
        ax.axis("on")
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.grid(linewidth=0.4, linestyle="--", dashes=(5, 20))
        if cfg.small_scale:
            scalex = 1e-10
            if cfg.plot_prob:
                scaley = 1e10
            else:
                scaley = 1e5
        else:
            scalex = 1
            scaley = 1

        if p.potential == 1 or p.potential == 2:
            xlimd = -10 * scalex
            xlimu = 10 * scalex
        else:
            xlimd = -5 * scalex
            xlimu = 15 * scalex
        if cfg.plot_prob:
            ylimd = -0.1 * scaley
            ylimu = 1.0 * scaley
            y_xi = -0.05 * scaley
        else:
            if cfg.plot_phase:
                ylimd = -0.1 * scaley
                y_xi = -0.05 * scaley
            else:
                ylimd = -1.0 * scaley
                y_xi = -0.1 * scaley
            ylimu = 1.0 * scaley
        if not cfg.small_scale:
            ax.text(xlimu, y_xi, '$\\xi$', fontsize=18)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        ax.set_xlim(xlimd, xlimu)
        ax.set_ylim(ylimd, ylimu)

    def create_barrier(self, ax):
        color = (0.83, 0.83, 0.83)
        if cfg.small_scale:
            scalex = 1e-10
            if cfg.plot_prob:
                scaley = 1e10
            else:
                scaley = 1e5
        else:
            scalex = 1
            scaley = 1
        match p.potential:
            case 0:
                ax.plot(self._x, self._V,
                        color="k", linestyle="-", linewidth=1)
                y1 = self._V - 0.2 * scaley
                ax.fill_between(self._x, self._V, y1,
                                where=(self._V > y1), color=color)
            case 1:
                if cfg.small_scale:
                    if cfg.plot_prob:
                        scale2 = 1e27
                    else:
                        scale2 = 1e22
                else:
                    scale2 = 0.05
                # spread the potential for visualization
                V = self._V * scale2
                ax.plot(self._x, V, color="k", linestyle="-", linewidth=1)
                # Calculate total initial energy
                E = p.p**2 / (2 * p.m) + 0.5 * p.m * p.omega**2 * p.x0**2
                # Calculate turning points
                x_turn = np.sqrt(2 * E / (p.m * p.omega**2))
                if cfg.verbose:
                    print(f"inversion point ±{x_turn}")
                ax.plot([-x_turn, -x_turn], [0, 1000 * scaley], color='k',
                        linestyle='--', dashes=(10, 10), linewidth=0.7)
                ax.plot([x_turn, x_turn], [0, 10001000 * scaley], color='k',
                        linestyle='--', dashes=(10, 10), linewidth=0.7)
                if cfg.small_scale:
                    if cfg.plot_prob:
                        y1 = (V - 1e9) - 0.1 * self._x**2 * scale2
                    else:
                        y1 = (V - 2e4) - 0.1 * self._x**2 * scale2
                else:
                    y1 = (self._V - 5) / 20 - 0.1 * self._x**2 / 20
                ax.fill_between(self._x, V, y1, where=(V > y1), color=color)
            case 2:
                px = np.array([-p.Vx_bar, -p.Vx_bar, p.Vx_bar, p.Vx_bar])
                py = np.array([1e12, 0, 0, 1e12])
                ax.plot(px, py, color="k", linestyle="-", linewidth=1)
                px = np.array([-p.Vx_bar - 1 * scalex, -p.Vx_bar])
                py = np.array([1e12, 1e12])
                y1 = np.array([-0.2 * scaley, -0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
                px = np.array([p.Vx_bar, p.Vx_bar + 1 * scalex])
                py = np.array([1e12, 1e12])
                y1 = np.array([-0.2 * scaley, -0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
                px = np.array([-p.Vx_bar, p.Vx_bar])
                py = np.array([0, 0])
                y1 = np.array([-0.2 * scaley, -0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
            case 3:
                if cfg.small_scale:
                    if cfg.plot_prob:
                        scale2 = 0.5e28
                    else:
                        scale2 = 0.5e23
                else:
                    scale2 = 0.5
                # shrink the potential for visualization
                V = p.V_barrier * scale2
                px = np.array([-p.x_max, p.Vx_bar, p.Vx_bar, p.x_max])
                py = np.array([0, 0, V, V])
                ax.plot(px, py, color="k", linestyle="-", linewidth=1)
                px = np.array([p.Vx_bar, p.Vx_bar + 1.2 * scalex])
                py = np.array([V, V])
                y1 = np.array([-0.2 * scaley, -0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
                px = np.array([-p.x_max, p.Vx_bar])
                py = np.array([0, 0])
                y1 = np.array([-0.2 * scaley, -0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
                px = np.array([p.Vx_bar, p.x_max])
                py = np.array([V, V])
                y1 = np.array([V - 0.2 * scaley, V - 0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
            case 4:
                if cfg.small_scale:
                    if cfg.plot_prob:
                        scale2 = 0.5e28
                    else:
                        scale2 = 0.5e23
                else:
                    scale2 = 0.5
                # shrink the potential for visualization
                V = p.V_barrier * scale2
                px = np.array([-p.x_max, p.Vx_bar, p.Vx_bar,
                               p.Vx_finite_bar, p.Vx_finite_bar, p.x_max])
                py = np.array([0, 0, V, V, 0, 0])
                ax.plot(px, py, color="k", linestyle="-", linewidth=1)
                px = np.array([p.Vx_bar, p.Vx_bar + 1.2 * scalex])
                py = np.array([V, V])
                y1 = np.array([0, 0])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
                px = np.array([-p.x_max, p.Vx_bar])
                py = np.array([0, 0])
                y1 = np.array([-0.2 * scaley, -0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)
                px = np.array([p.Vx_finite_bar, p.x_max])
                py = np.array([0, 0])
                y1 = np.array([-0.2 * scaley, -0.2 * scaley])
                ax.fill_between(px, py, y1, where=(py > y1), color=color)

        if p.potential == 3 or p.potential == 4:
            T = T_wavepacket(self._psi)
            if cfg.verbose:
                print(f"momentum of the wavepacket {T}")
            if cfg.small_scale:
                latex_str = (
                    f"$\\begin{{array}}{{rl}} V_{{0}} & = {p.V_barrier:.2e}"
                    + f"\\\\ \\langle p \\rangle & = {T:.2e} \\end{{array}}$")
            else:
                latex_str = (
                    f"$\\begin{{array}}{{rl}} V_{{0}} & = {p.V_barrier}"
                    + f"\\\\ \\langle p \\rangle & = {T:.2f} \\end{{array}}$")
            ax.text(0.05, 0.85, latex_str, transform=ax.transAxes,
                    ha='left', va='center', fontsize=20)

    def init_graph(self, ax):
        lim_td = 0.0
        lim_tu = p.t_max
        self._t = np.arange(lim_td, lim_tu, p.dt)
        # adjust the simulation step necessary for each iteration so that there
        # are the necessary number of seconds in the animation
        self._step = len(self._t) // self._total_frames + 1
        self._x = np.arange(-p.x_max, p.x_max, p.dx)
        if cfg.verbose:
            print(f"number of total time steps {len(self._t)}")
            print(f"number of total space steps {len(self._x)}")
            print(f"number of time steps for each frame {self._step}")
        if cfg.superposition:
            self._psi = create_superposition(self._x)
        else:
            self._psi = create_wavepacket(self._x)
        N = len(self._x)
        # define the potential and plot it
        self._V = create_potential(self._x)
        self.create_barrier(ax)
        if cfg.increase_precision:
            # create potential energy matrix
            V = sparse.diags(self._V, 0)
            # create kinetic energy matrix
            T = sparse.diags([1, -2, 1], [-1, 0, 1],
                             shape=(N, N)) * (-p.hbar**2 / (2 * p.m * p.dx**2))
            # Hamiltonian matrix (kinetic + potential)
            self._H = T + V
        else:
            # if scipy is not used, allocate the dense matrices for the matrix
            # multiplication
            self._H = np.zeros((N, N), dtype=complex)
            for i in range(N):
                self._H[i, i] = self._V[i]
                if i > 0:
                    self._H[i, i - 1] = -p.hbar**2 / (2 * p.m * p.dx**2)
                if i < N - 1:
                    self._H[i, i + 1] = -p.hbar**2 / (2 * p.m * p.dx**2)
            # Create identity matrix
            In = np.eye(N, dtype=complex)

            # Time evolution operators (Crank-Nicolson method)
            factor = 1j * p.dt / (2 * p.hbar)
            self._A = In - factor * self._H
            self._B = In + factor * self._H

        # initialize the plots and store the line objects
        if cfg.plot_prob:
            self._line, = ax.plot([], [], lw=2, color=c.r, label='$|\\Psi|^2$')
            plt.legend(handles=[self._line], fontsize=24)
        else:
            if cfg.plot_phase:
                self._poly = PolyCollection([], cmap='hsv', edgecolors='none')
                ax.add_collection(self._poly)
            else:
                self._line1, = ax.plot(
                    [], [], lw=1.5, color=c.b, label='$\\Re\\{\\Psi\\}$')
                self._line2, = ax.plot(
                    [], [], lw=1.5, color=c.o, label='$\\Im\\{\\Psi\\}$')
            self._line3, = ax.plot([], [], lw=2, color=c.g, label='$|\\Psi|$')
            if cfg.plot_phase:
                plt.legend(handles=[self._line3], fontsize=24)
            else:
                plt.legend(
                    handles=[self._line3, self._line2, self._line1],
                    fontsize=24)
        self._text_obj = ax.text(
            0.7, 0.85, '', transform=ax.transAxes,
            ha='center', va='center', fontsize=20)

        # make the starting plot
        self.plot_update(0)

    def plot(self):
        if self.do_plot:
            self._fig_p, self._axs_p = plt.subplots(figsize=(12, 8), dpi=300)
            if self.do_plot:
                super().plot()
                self.create_axes(True)
                self.init_graph(self._axs_p)
                # Select with frame to plot without the need of animation
                # frame_to_plot = self._total_frames - 1
                frame_to_plot = 0
                for i in range(frame_to_plot):
                    self.frame_update(i, self._axs_p, True)

    def init_animation(self):
        if not self.do_animation:
            return
        self._fig_a, self._axs_a = plt.subplots(figsize=(12, 8), dpi=100)
        super().init_animation()
        self.create_axes(False)
        self.init_graph(self._axs_a)

    def animate(self):
        if not self.do_animation:
            return
        anim_args = (self._axs_a, False)
        self.test_animation_frame = 30
        super().animate(self.frame_update, self._total_frames, anim_args)

    def frame_update(self, i: int, axs: np.ndarray, is_plot: bool):
        # if already after the end of the time to simulate return
        if self._step * p.dt * i > self._t[-1]:
            if not hasattr(self, '_end_warning'):
                self._end_warning = True
                print(f"Simulation ended at frame {i}")
            return
        if not is_plot:
            super().frame_update(i)
        if cfg.verbose:
            print(f"current frame: {i+1}")
        # advance time evolution
        if cfg.increase_precision:
            # define the time span and evaluation points for do many steps
            # with a single calculation
            t_span = [0, self._step * p.dt]
            t_eval = np.linspace(0, self._step * p.dt, self._step + 1)
            sol = integrate.solve_ivp(
                schrodinger_rhs, t_span, self._psi, method=cfg.rk_method,
                t_eval=t_eval, args=(self._H,))
            self._psi = sol.y[:, -1]  # update psi to the last computed value
        else:
            for _ in range(self._step):
                self._psi = np.linalg.solve(self._A, (self._B @ self._psi))
        self.plot_update(i + 1)

    def plot_update(self, cur_step: int):
        if cfg.plot_prob:
            self._line.set_data(
                self._x, abs(self._psi.conjugate() * self._psi))
        else:
            if cfg.plot_phase:
                # extract the phase of psi
                phase = np.angle(self._psi)
                # normalize phase to [0, 1] for hsl
                normalized_phase = (phase + np.pi) / (2 * np.pi)
                # create vertices for PolyCollection
                verts = [np.column_stack([self._x, np.zeros_like(self._x)])]
                verts.append(np.column_stack([self._x, abs(self._psi)]))
                # update the PolyCollection with phase colors
                polys = [np.column_stack(
                    [[self._x[j], self._x[j + 1], self._x[j + 1], self._x[j]],
                     [0, 0, abs(self._psi[j + 1]), abs(self._psi[j])]])
                    for j in range(len(self._x) - 1)]
                self._poly.set_verts(polys)
                self._poly.set_array(normalized_phase[:-1])
            else:
                self._line1.set_data(self._x, self._psi.real)
                self._line2.set_data(self._x, self._psi.imag)
            self._line3.set_data(self._x, abs(self._psi))
        # take into account minor rounding on the last iteration
        t_end = self._step * p.dt * cur_step
        if cfg.compute_prob:
            # compute the probability density
            prob = integrate.simps(
                np.abs(self._psi.conj() * self._psi), self._x)
            if cfg.small_scale:
                formatted_text = (
                    f"$\\begin{{array}}{{rl}} t & = {t_end:.2e} \\\\"
                    + f"P & = {prob:.2f} \\end{{array}}$")
            else:
                formatted_text = (
                    f"$\\begin{{array}}{{rl}} t & = {t_end:.2f} \\\\"
                    + f"P & = {prob:.2f} \\end{{array}}$")
        else:
            if cfg.small_scale:
                formatted_text = (f'$t={t_end:.2e}$')
            else:
                formatted_text = (f'$t={t_end:.2f}$')
        self._text_obj.set_text(formatted_text)


def make_plot(outfile: str):
    params = {
        'high_res': True,
        'ggplot': False,
        'dark_background': False,
        'do_plot': True,
        'do_animation': True,
        'animation_format': 'gif',
        'total_duration': 6,
        'fps': 30
    }
    plotter = MyPlotter(params, outfile)
    plotter.plot()
    plotter.save_plot()
    plotter.init_animation()
    plotter.animate()
    plotter.save_animation()


def main():
    parser = argparse.ArgumentParser(
        description='schrodinger 1d simulation')
    parser.add_argument('-o', '--ofile', help='output file')
    args = parser.parse_args()
    if args.ofile:
        ofile = args.ofile
    else:
        ofile = "schrodinger_1d.png"
    make_plot(ofile)


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
