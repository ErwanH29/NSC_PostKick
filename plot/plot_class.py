import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

class SetupFig(object):
    def __init__(self):
        self.TICK_SIZE = 14
        cmap = plt.colormaps["cool"]
        self.cmap = cmap(np.linspace(0.15, 1, 6))
        self.bin_colours = ["tab:red", "tab:blue"]
        self.colours = ["tab:blue", "tab:red", "black", "tab:red"]
        self.linestyles = ['-', ':']
    
    def _hist_tickers(self, ax):
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.tick_params(axis="y", labelsize=self.TICK_SIZE)
        ax.tick_params(axis="x", labelsize=self.TICK_SIZE)
        return ax
    
    def _tickers(self, ax):
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.tick_params(
            axis="y", which='both', 
            direction="in",
            labelsize=self.TICK_SIZE
        )
        ax.tick_params(axis="x", which='both',
            direction="in",
            labelsize=self.TICK_SIZE
        )
        return ax
    
    def get_fig_ax(self, figsize=(6,5), ptype="standard"):
        fig, ax = plt.subplots(figsize=figsize)
        if ptype=="hist":
            ax = self._hist_tickers(ax)
        else:
            ax = self._tickers(ax)
        return fig, ax
    
    def two_panel_fig(self, figsize=(8,6)):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 2,  width_ratios=(4,2), height_ratios=(2,5),
            left=0.1, right=0.9, bottom=0.1, top=0.9,
            wspace=0.05, hspace=0.05
            )
        ax = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
        for ax_ in [ax, ax1]:
            self._tickers(ax_)
            
        return fig, [ax, ax1]
    
    def moving_average(self, array, smoothing):
        """
        Conduct running average of some variable
        Args:
            array (list):       Array hosting values
            smoothing (float):  Number of elements to average over
        Returns: List of values smoothed over some length
        """
        value = np.cumsum(array, dtype=float)
        value[smoothing:] = value[smoothing:] - value[:-smoothing]
        return value[smoothing-1:]/smoothing

