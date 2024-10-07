#!/usr/bin/env python3
'''
/************************/
/*  mod_plotter_1d.py   */
/*     Version 1.0      */
/*      2024/06/02      */
/************************/
'''
from abc import ABC
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
import time


class BasePlotter(ABC):
    def __init__(self, params: dict, outfile: str):
        plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} " \
            r"\usepackage{amsmath} \usepackage{helvet}"
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica",
            "font.sans-serif": "Helvetica"
        })
        plt.rcParams['animation.convert_path'] = 'magick'
        if params.get('use_ggplot'):
            plt.style.use('ggplot')
        if params.get('dark_background'):
            plt.style.use('dark_background')
        self._high_res = params.get('high_res', False)
        self._do_plot = params.get('do_plot', False)
        self._do_animation = params.get('do_animation', False)
        self._test_animation = params.get('test_animation', False)
        self._test_animation_frame = 0
        self._total_duration = params.get('total_duration', 25)
        self._fps = params.get('fps', 120)
        self._total_frames = self._total_duration * self._fps
        self._interval = 1000 / self._fps
        self._start_time = time.time()
        self._last_printed_second = 0
        self._outfile_p = outfile
        base, ext = self._outfile_p.rsplit('.', 1)
        self._animation_format = params.get('animation_format', 'mp4')
        self._outfile_a = f"{base}.{self._animation_format}"
        self._outfile_at = f"{base}_a_test.{ext}"
        self._fig_p = None
        self._fig_a = None
        self._anim = None
        self._axs_p = None
        self._axs_a = None

    @property
    def high_res(self):
        return self._high_res

    @high_res.setter
    def high_res(self, value: bool):
        self._high_res = value

    @property
    def do_plot(self):
        return self._do_plot

    @do_plot.setter
    def do_plot(self, value: bool):
        self._do_plot = value

    @property
    def do_animation(self):
        return self._do_animation

    @do_animation.setter
    def do_animation(self, value: bool):
        self._do_animation = value

    @property
    def test_animation(self):
        return self._test_animation

    @test_animation.setter
    def test_animation(self, value: bool):
        self._test_animation = value

    @property
    def test_animation_frame(self):
        return self._test_animation_frame

    @test_animation_frame.setter
    def test_animation_frame(self, value: int):
        self._test_animation_frame = value

    @property
    def total_duration(self):
        return self._total_duration

    @total_duration.setter
    def total_duration(self, value: float):
        self._total_duration = value
        self._total_frames = self._total_duration * \
            self._fps  # Update total_frames accordingly

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value: int):
        self._fps = value
        self._interval = 1000 / self._fps  # Update interval accordingly
        self._total_frames = self._total_duration * \
            self._fps  # Update total_frames accordingly

    @property
    def total_frames(self):
        return self._total_frames

    @property
    def interval(self):
        return self._interval

    @property
    def start_time(self):
        return self._start_time

    @property
    def last_printed_second(self):
        return self._last_printed_second

    @last_printed_second.setter
    def last_printed_second(self, value):
        self._last_printed_second = value

    def create_axes(self, is_plot: bool):
        pass

    def plot(self):
        if self._do_plot and not self._fig_p:
            if self._high_res:
                # For 4K resolution at 300 DPI
                dpi = 300
                figsize = (3840 / dpi, 2160 / dpi)
            else:
                # For HD resolution at 100 DPI
                dpi = 100
                figsize = (1920 / dpi, 1080 / dpi)
            self._fig_p, self._axs_p = plt.subplots(
                1, 1, figsize=figsize, dpi=dpi)

    def init_animation(self):
        if self._do_animation and not self._fig_a:
            if self._high_res:
                # For 4K resolution at 300 DPI
                dpi = 300
                figsize = (3840 / dpi, 2160 / dpi)
            else:
                # For HD resolution at 100 DPI
                dpi = 100
                figsize = (1920 / dpi, 1080 / dpi)
            self._fig_a, self._axs_a = plt.subplots(
                1, 1, figsize=figsize, dpi=dpi)

    def animate(self, frame_update, frames, fargs):
        # set here to ensure the axis are already labeled
        self._fig_a.tight_layout()
        if self._test_animation:
            frame_update(self._test_animation_frame, *fargs)
        else:
            self._anim = FuncAnimation(
                self._fig_a, frame_update, frames=frames,
                fargs=fargs, interval=self._interval, repeat=True)

    def frame_update(self, i: int):
        # Calculate the current time in seconds
        current_time = i / self._fps
        # Check if a new second has been reached
        if current_time >= self._last_printed_second + 1:
            # Calculate the elapsed time
            elapsed_time = time.time() - self._start_time
            duration = timedelta(seconds=int(elapsed_time))
            if duration.total_seconds() >= 3600:
                elapsed_time_formatted = str(duration)
            else:
                minutes, seconds = divmod(duration.total_seconds(), 60)
                elapsed_time_formatted = (
                    f"{int(minutes):02d}:{int(seconds):02d}")
            self._last_printed_second = int(current_time)
            print(
                f"Encoded {self._last_printed_second} "
                f"second{'s' if self._last_printed_second > 1 else ''}, "
                f"Elapsed Time: {elapsed_time_formatted}"
            )

    def save_plot(self):
        if self._fig_p:
            # set here to ensure the axis are already labeled
            self._fig_p.tight_layout()
            self._fig_p.savefig(self._outfile_p)
            print(f"Plot saved to {self._outfile_p}.")

    def save_animation(self):
        if self._test_animation:
            if self._fig_a:
                self._fig_a.savefig(self._outfile_at)
                print(f"Test plot saved to {self._outfile_at}.")
        else:
            if self._anim:
                if self._animation_format == 'mp4':
                    self._anim.save(self._outfile_a, writer='ffmpeg')
                elif self._animation_format == 'gif':
                    self._anim.save(self._outfile_a, writer='imagemagick')
                elif self._animation_format == 'html':
                    self._anim.save(self._outfile_a, writer='html')
                elif self._animation_format == 'jshtml':
                    outfile_a = os.path.splitext(self._outfile_a)[0] + '.html'
                    with open(outfile_a, 'w') as f:
                        f.write(self._anim.to_jshtml(
                            fps=self._fps, default_mode='loop'))
                else:
                    raise ValueError("Unsupported animation format. "
                                     "Use 'mp4', 'gif', 'html' or 'jshtml'.")
                print(f"Animation saved to {self._outfile_a}.")


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise RuntimeError('Must be using Python 3')
    pass
