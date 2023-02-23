import importlib
import datetime

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import global_variables
import lxi_misc_codes as lmsc

importlib.reload(global_variables)
importlib.reload(lmsc)


class plot_data_class():
    """
    A class for plotting different kinds of data

    Attributes:
        df_slice_hk: pandas dataframe
            The dataframe containing the Housekeeping data. Once you upload an HK file to the GUI
            the dataframe is stored in a global variable called
            global_variables.all_file_details["df_slice_hk"] and corresponds to the value in the
            file which was loaded.
        df_slice_sci: pandas dataframe
            The dataframe containing the Science data. Once you upload an SCI file to the GUI
            the dataframe is stored in a global variable called
            global_variables.all_file_details["df_slice_sci"] and corresponds to the value in the
            file which was loaded.
        start_time: str
            The start time of the data to be plotted. By default this is the first time in the
            dataframe.
        end_time: str
            The end time of the data to be plotted. By default this is the last time in the
            dataframe.
        plot_key: str
            The key in the dataframe to be plotted for the time series of the housekeeping data.
        channel1: str
            The channel to be plotted along x-axis for the histogram of the voltage data from the
            science data. Though the default value is set to None here, when you start the GUI, the
            value of "channel1" is set to either "Channel1" or "Channel3" depending on which
            one is plotting in the GUI.
        channel2: str
            The channel to be plotted along y-axis for the histogram of the voltage data from the
            science data. Though the default value is set to None here, when you start the GUI, the
            value of "channel2" is set to either "Channel2" or "Channel4" depending on which
            one is plotting in the GUI.
        bins: int
            The number of bins to be used for the histogram of the main plot, the one that shows the
            distribution of photons on the (x,y) plane.
        cmin: int
            The minimum value of the colorbar for the main plot. Default is 1.
        cmax: int
            The maximum value of the colorbar for the main plot. Be careful while using it, since
            anything above this value will be clipped. Default is None.
        x_min: float
            The minimum value of the x-axis for the main plot. Default is minimum value of x in the
            science data.
        x_max: float
            The maximum value of the x-axis for the main plot. Default is maximum value of x in the
            science data.
        y_min: float
            The minimum value of the y-axis for the main plot. Default is minimum value of y in the
            science data.
        y_max: float
            The maximum value of the y-axis for the main plot. Default is maximum value of y in the
            science data.
        density: bool
            Whether to plot the histogram as a density or not. Default is False.
        norm: bool
            The scale of colorbar to be plotted. Options are "log" or "linear". Default is "log".
        ts_fig_height: float
            The height of the time series plot. Default is 6.
        ts_fig_width: float
            The width of the time series plot. Default is 12.
        hist_fig_height: float
            The height of the histogram plot. Default is 6.
        hist_fig_width: float
            The width of the histogram plot. Default is 12.
        volt_fig_height: float
            The height of the voltage plot. Default is 6.
        volt_fig_width: float
            The width of the voltage plot. Default is 12.

    Methods:
        ts_plots:
            Plots the time series of any given parameter from the housekeeping data. It is a
            initialized by the __init__ method. It takes the following arguments:
                - self: The object itself.
                - plot_key: The key in the dataframe to be plotted for the time series of the
                            housekeeping data.
                - start_time: The start time of the data to be plotted. By default this is the
                                first time in the dataframe.
                - end_time: The end time of the data to be plotted. By default this is the last
                                time in the dataframe.

        hist_plots:
            Plots the histogram of the x and y position of the observation. It is a initialized by
            the __init__ method. It needs the following arguments to be passed:
                - self: The object itself.
                - t_start: The start time of the data to be plotted. By default this is the first
                            time in the dataframe.
                - t_end: The end time of the data to be plotted. By default this is the last time
                            in the dataframe.
                - bins: The number of bins to be used for the histogram of the main plot, the one
                            that shows the distribution of photons on the (x,y) plane.
                - cmin: The minimum value of the colorbar for the main plot. Default is 1.
                - cmax: The maximum value of the colorbar for the main plot. Be careful while
                            using it, since anything above this value will be clipped. Default is
                            None.
                - x_min: The minimum value of the x-axis for the main plot. Default is minimum
                            value of x in the science data.
                - x_max: The maximum value of the x-axis for the main plot. Default is maximum
                            value of x in the science data.
                - y_min: The minimum value of the y-axis for the main plot. Default is minimum
                            value of y in the science data.
                - y_max: The maximum value of the y-axis for the main plot. Default is maximum
                            value of y in the science data.
                - density: Whether to plot the histogram as a density or not. Default is False.
                - norm: The scale of colorbar to be plotted. Options are "log" or "linear".
                            Default is "log".

        hist_plots_volt:
            Plots the histogram of the voltage of the observation. It is a initialized by the
            __init__ method. It needs the following arguments to be passed:
                - self: The object itself.
                - t_start: The start time of the data to be plotted. By default this is the first
                            time in the dataframe.
                - t_end: The end time of the data to be plotted. By default this is the last time
                            in the dataframe.
                - channel1: The channel to be plotted along x-axis for the histogram of the voltage
                            data from the science data. Though the default value is set to None
                            here, when you start the GUI, the value of "channel1" is set to either
                            "Channel1" or "Channel3" depending on which one is plotting in the
                            GUI.
                - channel2: The channel to be plotted along y-axis for the histogram of the voltage
                            data from the science data. Though the default value is set to None
                            here, when you start the GUI, the value of "channel2" is set to either
                            "Channel2" or "Channel4" depending on which one is plotting in the
                            GUI.
    """

    def __init__(self,
                 df_slice_hk=None,
                 df_slice_sci=None,
                 start_time=None,
                 end_time=None,
                 plot_key=None,
                 channel1=None,
                 channel2=None,
                 bins=None,
                 cmin=None,
                 cmax=None,
                 x_min=None,
                 x_max=None,
                 y_min=None,
                 y_max=None,
                 density=None,
                 norm=None,
                 unit=None,
                 ts_fig_height=None,
                 ts_fig_width=None,
                 hist_fig_height=None,
                 hist_fig_width=None,
                 volt_fig_height=None,
                 volt_fig_width=None,
                 v_min=None,
                 v_max=None,
                 v_sum_min=None,
                 v_sum_max=None,
                 cut_status_var=None,
                 crv_fit=None,
                 lin_corr=None,
                 cmap=None,
                 use_fig_size=None,
                 ):
        self.df_slice_hk = df_slice_hk
        self.df_slice_sci = df_slice_sci
        self.start_time = start_time
        self.end_time = end_time
        self.plot_key = plot_key
        self.channel1 = channel1
        self.channel2 = channel2
        self.bins = bins
        self.cmin = cmin
        self.cmax = cmax
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.density = density
        self.norm = norm
        self.unit = unit
        self.ts_fig_height = ts_fig_height
        self.ts_fig_width = ts_fig_width
        self.hist_fig_height = hist_fig_height
        self.hist_fig_width = hist_fig_width
        self.volt_fig_height = volt_fig_height
        self.volt_fig_width = volt_fig_width
        self.v_min = v_min
        self.v_max = v_max
        self.v_sum_min = v_sum_min
        self.v_sum_max = v_sum_max
        self.cut_status_var = cut_status_var
        self.crv_fit = crv_fit
        self.lin_corr = lin_corr
        self.cmap = cmap
        self.use_fig_size = use_fig_size

    def ts_plots(self):
        """
        Plot the time series of the data

        Return
        ------
            fig: figure object
        """
        # Try to convert the start_time and end_time to float or int
        try:
            t_start = datetime.datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            t_start = self.df_slice_hk.index.min()
            pass
        try:
            t_end = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            t_end = self.df_slice_hk.index.max()
            pass

        # Make a dictionary of all the plot options and their units
        unit_dict = {"HK_id": "#",
                     "PinPullerTemp": "(K)",
                     "OpticsTemp": "(K)",
                     "LEXIbaseTemp": "(C)",
                     "HVsupplyTemp": "(K)",
                     "+5.2V_Imon": "(mA)",
                     "+10V_Imon": "(mA)",
                     "+3.3V_Imon": "(mA)",
                     "AnodeVoltMon": "(V)",
                     "+28V_Imon": "(mA)",
                     "ADC_Ground": "(V)",
                     "Cmd_count": "#",
                     "Pinpuller_Armed": "",
                     "HVmcpAuto": "",
                     "HVmcpMan": "",
                     "DeltaEvntCount": "#",
                     "DeltaDroppedCount": "#",
                     "DeltaLostevntCount": "#"
                     }

        alpha = 0.4
        ms = 2
        # Plot the data
        fig = plt.figure(num=None, figsize=(self.ts_fig_width * 1.5, self.ts_fig_height),
                         edgecolor='k', facecolor='w')
        fig.subplots_adjust(left=0.25, right=0.99, top=0.99, bottom=0.25, wspace=0, hspace=0)
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1])

        # Set the df_slice_hk to the time range specified by the user in the GUI and plot it
        self.df_slice_hk = self.df_slice_hk.loc[t_start:t_end]
        x_axs_val = (self.df_slice_hk.index - t_start).total_seconds()
        axs1 = plt.subplot(gs[:])
        axs1.plot(x_axs_val, self.df_slice_hk[self.plot_key], '.', color="darkred",
                  alpha=alpha, ms=ms, label=self.plot_key)
        axs1.set_xlim(np.nanmin(x_axs_val), np.nanmax(x_axs_val))
        # Rotate the x-axis labels by certain degrees and set their fontsize, if required
        plt.setp(axs1.get_xticklabels(), rotation=0)
        axs1.set_xlabel(f'Time since {t_start} (s)')
        axs1.set_ylabel(f"{unit_dict[self.plot_key]}")
        axs1.tick_params(axis="both", which="major")
        axs1.legend(loc='best')
        legend_list = axs1.legend(handlelength=0, handletextpad=0, fancybox=False)
        for item in legend_list.legendHandles:
            item.set_visible(False)

        plt.close("all")
        return fig

    def hist_plots(self):
        """
        Plot the histogram of the data

        Return
        ------
            fig: figure object
        """

        # Try to convert the start_time and end_time to float or int
        try:
            t_start = datetime.datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            t_start = self.df_slice_sci.index.min()
            pass
        try:
            t_end = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            t_end = self.df_slice_sci.index.max()
            pass
        try:
            bins = int(self.bins)
        except Exception:
            bins = 50
        try:
            cmin = float(self.cmin)
        except Exception:
            cmin = 1
        try:
            cmax = float(self.cmax)
        except Exception:
            cmax = None
        try:
            x_min = float(self.x_min)
        except Exception:
            x_min = self.df_slice_sci["x_val"].min()
        try:
            x_max = float(self.x_max)
        except Exception:
            x_max = self.df_slice_sci["x_val"].max()
        try:
            y_min = float(self.y_min)
        except Exception:
            y_min = self.df_slice_sci["y_val"].min()
        try:
            y_max = float(self.y_max)
        except Exception:
            y_max = self.df_slice_sci["y_val"].max()
        try:
            density = self.density
        except Exception:
            density = None
        try:
            v_min = float(self.v_min)
        except Exception:
            v_min = 0
        try:
            v_max = float(self.v_max)
        except Exception:
            v_max = 4
        try:
            v_sum_min = float(self.v_sum_min)
        except Exception:
            v_sum_min = 0
        try:
            v_sum_max = float(self.v_sum_max)
        except Exception:
            v_sum_max = 16

        # Check if norm is 'log' or 'linear'
        if (self.norm == 'log' or self.norm == 'linear'):
            norm = self.norm
        else:
            norm = None

        if norm == 'log':
            norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
        elif norm == 'linear':
            norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

        # Check whether the axes units are 'volt' or 'mcp'
        if (self.unit == 'volt' or self.unit == 'mcp'):
            unit = self.unit
        else:
            unit = 'mcp'

        # Check if cmap is an instance of mpl.colors.Colormap
        if self.cmap in mpl.pyplot.colormaps():
            self.cmap = self.cmap
        else:
            print(f"Invalid cmap: {self.cmap}. Using default cmap: 'viridis'")
            self.cmap = 'viridis'

        x_range = [x_min, x_max]
        y_range = [y_min, y_max]

        # Remove rows with duplicate indices
        self.df_slice_sci = self.df_slice_sci[~self.df_slice_sci.index.duplicated(keep='first')]
        # Select data in the specified time range
        self.df_slice_sci = self.df_slice_sci.loc[t_start:t_end]
        # Exclude channel1 to channel4 data based on v_min and v_max
        # Check if either v_min or v_max or v_sum_min or v_sum_max are None
        self.df_slice_sci = self.df_slice_sci[(self.df_slice_sci["Channel1"] >= v_min) &
                                              (self.df_slice_sci["Channel1"] <= v_max) &
                                              (self.df_slice_sci["Channel2"] >= v_min) &
                                              (self.df_slice_sci["Channel2"] <= v_max) &
                                              (self.df_slice_sci["Channel3"] >= v_min) &
                                              (self.df_slice_sci["Channel3"] <= v_max) &
                                              (self.df_slice_sci["Channel4"] >= v_min) &
                                              (self.df_slice_sci["Channel4"] <= v_max) &
                                              ((self.df_slice_sci["v1_shift"] +
                                                self.df_slice_sci["v2_shift"] +
                                                self.df_slice_sci["v3_shift"] +
                                                self.df_slice_sci["v4_shift"]) >= v_sum_min) &
                                              ((self.df_slice_sci["v1_shift"] +
                                                self.df_slice_sci["v2_shift"] +
                                                self.df_slice_sci["v3_shift"] +
                                                self.df_slice_sci["v4_shift"]) <= v_sum_max)]

        if self.use_fig_size:
            fig = plt.figure(num=None, figsize=(self.hist_fig_width, self.hist_fig_height),
                             facecolor='w', edgecolor='k')
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0., hspace=0.)
        else:
            fig = plt.figure(num=None, facecolor='k', edgecolor='k')

        # fig.subplots_adjust(wspace=0., hspace=0.1)
        gs = plt.GridSpec(21, 21)

        axs1 = fig.add_subplot(gs[:-3, 3:], aspect=1)

        # Drop all nans in the data
        self.df_slice_sci = self.df_slice_sci.dropna()
        # Try to select only rows where "IsCommanded" is False
        try:
            self.df_slice_sci = self.df_slice_sci[self.df_slice_sci["IsCommanded"] == False]
        except Exception:
            pass
        # Plot the histogram on axs1
        if self.lin_corr == False:
            if self.unit =='volt':
                x_key = "x_val"
                y_key = "y_val"
            elif self.unit == 'mcp':
                x_key = "x_mcp"
                y_key = "y_mcp"
        elif self.lin_corr == True:
            if self.unit =='volt':
                x_key = "x_val_lin"
                y_key = "y_val_lin"
            elif self.unit == 'mcp':
                x_key = "x_mcp_lin"
                y_key = "y_mcp_lin"

        print("\033[1;32m Plotting histogram with linearity correction set to "
              f"{self.lin_corr} and axes units set to {self.unit}\033[0m")
        counts, xedges, yedges, im = axs1.hist2d(self.df_slice_sci[x_key],
                                                 self.df_slice_sci[y_key], bins=bins,
                                                 cmap=self.cmap, norm=norm,
                                                 range=[x_range, y_range], cmin=cmin,
                                                 density=density)
        # Add histogram data detauls to the global dictionary
        if self.lin_corr == False:
            global_variables.data_org["counts"] = counts
            global_variables.data_org["xedges"] = xedges
            global_variables.data_org["yedges"] = yedges
        elif self.lin_corr == True:
            # Add histogram data details to the global dictionary
            global_variables.data_lin["counts"] = counts
            global_variables.data_lin["xedges"] = xedges
            global_variables.data_lin["yedges"] = yedges
        # Find the index of the maximum value in counts, ignoring NaNs
        max_index = np.unravel_index(np.nanargmax(counts, axis=None), counts.shape)

        if self.cut_status_var == True:
            # Draw a horizontal and vertical line at the maximum value
            axs1.axvline(x=(xedges[max_index[0]] + xedges[max_index[0] + 1]) / 2, color='k',
                         linestyle='--', linewidth=1, alpha=0.5)
            axs1.axhline(y=(yedges[max_index[1]] + yedges[max_index[1] + 1]) / 2, color='k',
                         linestyle='--', linewidth=1, alpha=0.5)

        # Show the minor ticks on the x and y axes
        axs1.minorticks_on()

        # Set the grid on for both axes for major and minor ticks
        axs1.grid(which='major', linestyle='--', linewidth='0.1', color='black', alpha=0.2)
        # axs1.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
        # Number of data points in each bin along the x- and y-axes
        yn = counts[max_index[0], :]
        xn = counts[:, max_index[1]]

        x_step = (xedges[1:] + xedges[0:-1]) / 2
        y_step = (yedges[1:] + yedges[0:-1]) / 2

        if self.cut_status_var == True:
            y_hist = fig.add_subplot(gs[1:-4, 0:3], sharey=axs1)
            x_hist = fig.add_subplot(gs[-3:-1, 3:], sharex=axs1)
            # Make step plot between xedges and xn
            x_hist.step(x_step, xn, color='k', where='post')
            x_hist.plot(xedges[1:], xn, 'ko', markerfacecolor='none', markeredgecolor='gray')
# 
            x_hist.set_xlabel('Vertical Cut')
            # Make step plot between yedges and yn
            y_hist.step(yn, y_step, color='k', where='post')
            y_hist.plot(yn, yedges[:-1], 'ko', markerfacecolor='none', markeredgecolor='gray')
            y_hist.invert_xaxis()
            y_hist.set_ylabel('Horizontal Cut')

        divider1 = make_axes_locatable(axs1)
        cax1 = divider1.append_axes("top", size="5%", pad=0.02)
        cbar1 = plt.colorbar(im, cax=cax1, orientation='horizontal', ticks=None, fraction=0.05,
                             pad=0.0)

        cbar1.ax.tick_params(axis='x', which='both', direction='in', labeltop=True, top=True,
                             labelbottom=False, bottom=False, width=1, length=10,
                             labelrotation=0, pad=0)

        cbar1.ax.xaxis.set_label_position('top')

        if density is True:
            cbar1.set_label('Density')
        else:
            cbar1.set_label('N', labelpad=0.0, rotation=0)

        # Put y-label and tickmarks on right side
        if self.cut_status_var == True:
            axs1.yaxis.tick_right()
            axs1.yaxis.set_label_position('right')
        else:
            axs1.yaxis.tick_left()
            axs1.yaxis.set_label_position('left')
        if self.unit == 'volt':
            x_label = 'Strip = V3/(V1+V3)'
            y_label = 'Wedge = V4/(V2+V4)'
        elif self.unit == 'mcp':
            x_label = 'X (cm)'
            y_label = 'Y (cm)'
        axs1.set_xlabel(x_label)
        axs1.set_ylabel(y_label)
        axs1.set_xlim(x_min, x_max)
        axs1.set_ylim(y_min, y_max)
        axs1.tick_params(axis="both", which="major")
        # Show ticks on both sides of the plot
        axs1.tick_params(axis='both', which='both', direction='in', left=True, right=True, top=True,
                         bottom=True)

        # If curve fit option is chosen, fit a Gaussian to the data and plot it
        if self.crv_fit and self.cut_status_var:
            try:
                from scipy.optimize import curve_fit
                x_vals = (xedges[max_index[0] - 10:max_index[0] + 10] +
                          xedges[max_index[0] - 9:max_index[0] + 11]) / 2
                y_vals = (yedges[max_index[1] - 10:max_index[1] + 10] +
                          yedges[max_index[1] - 9:max_index[1] + 11]) / 2
                x_vals_counts = yn[max_index[1] - 10:max_index[1] + 10]
                y_vals_counts = xn[max_index[0] - 10:max_index[0] + 10]

                # Replace NaNs with zeros
                x_vals_counts = x_vals_counts.astype(float)
                x_vals_counts[np.isnan(x_vals_counts)] = 0
                y_vals_counts = y_vals_counts.astype(float)
                y_vals_counts[np.isnan(y_vals_counts)] = 0

                # Fit a Gaussian to the data
                popt_x, _ = curve_fit(lmsc.curve_fit_func, x_vals, x_vals_counts)

                x_hist.plot(x_vals, lmsc.curve_fit_func(x_vals, *popt_x), 'r--')
                popt_y, _ = curve_fit(lmsc.curve_fit_func, y_vals, y_vals_counts)

                y_hist.plot(lmsc.curve_fit_func(y_vals, *popt_y), y_vals, 'r--')

                # Find the full width at half maximum of the fitted Gaussian
                x_fwhm = lmsc.fwhm(x_vals, lmsc.curve_fit_func(x_vals, *popt_x))
                y_fwhm = lmsc.fwhm(y_vals, lmsc.curve_fit_func(y_vals, *popt_y))
                # Print the fit values on the plot
                x_hist.text(0.05, 0.95, '$\mu$ = {:.3f}'.format(popt_x[1]),
                            transform=x_hist.transAxes, verticalalignment='top')
                x_hist.text(0.05, 0.70, '$\sigma$ = {:.3f}'.format(popt_x[2]),
                            transform=x_hist.transAxes, verticalalignment='top')
                x_hist.text(0.05, 0.45, '$FWHM$ = {:.3f}'.format(x_fwhm),
                            transform=x_hist.transAxes, verticalalignment='top')
                y_hist.text(0.05, 0.95, '$\mu$ = {:.3f}'.format(popt_y[1]),
                            transform=y_hist.transAxes, verticalalignment='top')
                y_hist.text(0.05, 0.90, '$\sigma$ = {:.3f}'.format(popt_y[2]),
                            transform=y_hist.transAxes, verticalalignment='top')
                y_hist.text(0.05, 0.85, '$FWHM$ = {:.3f}'.format(y_fwhm),
                            transform=y_hist.transAxes, verticalalignment='top')

            except Exception:
                print('Error: Could not fit Gaussian to data.')
                pass

        # Set tight layout
        axs1.set_aspect('equal', anchor="C")
        if self.cut_status_var:
            y_hist.set_aspect('auto', anchor="SW")
            x_hist.set_aspect('auto', anchor="C")

        # Save tge figure
        fig_format = "png"
        fig_name = f"../figures/lin_corr_{self.lin_corr}_unit_{self.unit}.{fig_format}"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight', format=fig_format, transparent=True)
        plt.close("all")

        # fig.tight_layout()
        return fig

    def hist_plots_volt(self):
        """
        This function creates a histogram of the voltage data.

        Return
        ------
            fig: figure object
        """
        # Try to convert the start_time and end_time to float or int
        try:
            t_start = datetime.datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            t_start = self.df_slice_sci.index.min()
            pass
        try:
            t_end = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            t_end = self.df_slice_sci.index.max()
            pass
        try:
            bins = int(self.bins)
        except Exception:
            bins = 50
        try:
            cmin = float(self.cmin)
        except Exception:
            cmin = 1
        try:
            cmax = float(self.cmax)
        except Exception:
            cmax = None
        # Check if norm is an instance of mpl.colors.Normalize
        if (self.norm == 'log' or self.norm == 'linear'):
            norm = self.norm
        else:
            norm = None
        try:
            density = self.density
        except Exception:
            density = None
        try:
            v_min = float(self.v_min)
        except Exception:
            v_min = None
        try:
            v_max = float(self.v_max)
        except Exception:
            v_max = None

        # Check if norm is an instance of mpl.colors.Normalize
        if (self.norm == 'log' or self.norm == 'linear'):
            norm = self.norm
        else:
            norm = None

        # If density is true, set cmin to None
        # if density is True:
        #     cmin = None

        if norm == 'log':
            norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
        elif norm == 'linear':
            norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)

        # Check if cmap is an instance of mpl.colors.Colormap
        if self.cmap in mpl.pyplot.colormaps():
            self.cmap = self.cmap
        else:
            self.cmap = 'viridis'

        self.df_slice_sci = self.df_slice_sci[~self.df_slice_sci.index.duplicated(keep='first')]

        # Exclude channel1 to channel4 data based on v_min and v_max
        if v_min is not None and v_max is not None:
            self.df_slice_sci = self.df_slice_sci[(self.df_slice_sci["Channel1"] >= v_min) &
                                                  (self.df_slice_sci["Channel1"] <= v_max) &
                                                  (self.df_slice_sci["Channel2"] >= v_min) &
                                                  (self.df_slice_sci["Channel2"] <= v_max) &
                                                  (self.df_slice_sci["Channel3"] >= v_min) &
                                                  (self.df_slice_sci["Channel3"] <= v_max) &
                                                  (self.df_slice_sci["Channel4"] >= v_min) &
                                                  (self.df_slice_sci["Channel4"] <= v_max)]

        # Select channel1 corresponding to the start_time and end_time
        self.df_slice_sci = self.df_slice_sci[(self.df_slice_sci.index >= t_start) &
                                              (self.df_slice_sci.index <= t_end)]
        v1 = self.df_slice_sci[self.channel1][
            (self.df_slice_sci.index >= t_start) & (self.df_slice_sci.index <= t_end)]
        v2 = self.df_slice_sci[self.channel2][
            (self.df_slice_sci.index >= t_start) & (self.df_slice_sci.index <= t_end)]

        fig = plt.figure(num=None, figsize=(self.volt_fig_width, self.volt_fig_height),
                         facecolor='w', edgecolor='k')

        x_range = [0.9 * np.nanmin(v1), 1.1 * np.nanmax(v1)]
        y_range = [0.9 * np.nanmin(v2), 1.1 * np.nanmax(v2)]

        gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1])
        axs1 = fig.add_subplot(gs[0, 0], aspect=1)
        _, _, _, im = axs1.hist2d(v1, v2, bins=bins, cmap=self.cmap, norm=norm,
                                  range=[x_range, y_range], cmin=cmin, density=density)
        divider1 = make_axes_locatable(axs1)
        cax1 = divider1.append_axes("top", size="5%", pad=0.01)
        cbar1 = plt.colorbar(im, cax=cax1, orientation='horizontal', ticks=None, fraction=0.05,
                             pad=0.0)

        cbar1.ax.tick_params(axis='x', which='both', direction='in', labeltop=True, top=True,
                             labelbottom=False, bottom=False,
                             labelrotation=0, pad=0)

        cbar1.ax.xaxis.set_label_position('top')

        axs1.set_xlabel(self.channel1)
        axs1.set_ylabel(self.channel2)

        plt.tight_layout()
        plt.close('all')

        return fig
