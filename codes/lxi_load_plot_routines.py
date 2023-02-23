import importlib
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import global_variables
import lxi_gui_plot_routines as lgpr
import lxi_file_read_funcs as lxrf

importlib.reload(lgpr)
importlib.reload(lxrf)
importlib.reload(global_variables)

global_variables.init()


def load_ts_plots(root=None, df_slice_hk=None, plot_key=None, start_time=None, end_time=None, row=2,
                  column=1, columnspan=2, rowspan=2, fig_width=None, fig_height=None):
    """
    Loads the time series plots for the selected time range and displays them in the GUI.

    Parameters
    ----------
    df_slice_hk : pandas.DataFrame
        The dataframe containing the HK data.
    plot_key : str
        The key of the HK data to be plotted.
    start_time : str
        The start time of the time range to be plotted.
    end_time : str
        The end time of the time range to be plotted.
    row : int
        The row in which the plots should be displayed.
    column : int
        The column in which the plots should be displayed.
    columnspan : int
        The number of columns the plots should span.
    rowspan : int
        The number of rows the plots should span.
    fig_width : float
        The width of the figure.
    fig_height : float
        The height of the figure.

    Returns
    -------
    None
    """
    # Set the fontstyle to Times New Roman
    font_mpl = {'family': 'serif', 'weight': 'normal'}
    plt.rc('font', **font_mpl)
    plt.rc('text', usetex=False)

    frame = tk.Frame(root)
    frame.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky='nsew')

    fig_ts = lgpr.plot_data_class(df_slice_hk=df_slice_hk, plot_key=plot_key, start_time=start_time,
                                  end_time=end_time, ts_fig_height=fig_height,
                                  ts_fig_width=fig_width).ts_plots()
    canvas = FigureCanvasTkAgg(fig_ts, master=frame)
    canvas.get_tk_widget().pack(side="left", fill='both', expand=False)
    canvas.draw()


def load_hist_plots(root=None, df_slice_sci=None, start_time=None, end_time=None, bins=None,
                    cmin=None, cmax=None, x_min=None, x_max=None, y_min=None, y_max=None,
                    density=None, norm=None, unit=None, row=3, column=1, fig_width=5, fig_height=5,
                    columnspan=2, rowspan=2, v_min=2.2, v_max=3.9, v_sum_min=3, v_sum_max=12,
                    cut_status_var=False, crv_fit=False, lin_corr=False, cmap=None,
                    use_fig_size=False):
    """
    Loads the histogram plots for the selected time range and displays them in the GUI.

    Parameters
    ----------
    df_slice_sci : pandas.DataFrame
        The dataframe containing the science data.
    start_time : str
        The start time of the time range to be plotted.
    end_time : str
        The end time of the time range to be plotted.
    bins : int
        The number of bins to be used for the histogram.
    cmin : int
        The minimum value of the colorbar.
    cmax : int
        The maximum value of the colorbar.
    x_min : int
        The minimum value of the x-axis.
    x_max : int
        The maximum value of the x-axis.
    y_min : int
        The minimum value of the y-axis.
    y_max : int
        The maximum value of the y-axis.
    density : bool
        Whether or not the histogram should be normalized.
    norm : bool
        Whether or not the histogram should be normalized.
    unit : str
        The unit of the two axes
    row : int
        The row in which the plots should be displayed.
    column : int
        The column in which the plots should be displayed.
    fig_width : float
        The width of the figure.
    fig_height : float
        The height of the figure.
    columnspan : int
        The number of columns the plots should span.
    rowspan : int
        The number of rows the plots should span.

    Returns
    -------
    None
    """
    # Set the fontstyle to Times New Roman
    font_mpl = {'family': 'serif', 'weight': 'normal'}
    plt.rc('font', **font_mpl)
    plt.rc('text', usetex=False)

    fig_hist = lgpr.plot_data_class(df_slice_sci=df_slice_sci, start_time=start_time,
                                    end_time=end_time, bins=bins, cmin=cmin, cmax=cmax,
                                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                    density=density, norm=norm, unit=unit, 
                                    hist_fig_height=fig_height, hist_fig_width=fig_width,
                                    v_min=v_min, v_max=v_max, v_sum_min=v_sum_min,
                                    v_sum_max=v_sum_max, cut_status_var=cut_status_var,
                                    crv_fit=crv_fit, lin_corr=lin_corr, cmap=cmap,
                                    use_fig_size=use_fig_size).hist_plots()
    frame = tk.Frame(root)
    frame.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky='nsew')
    canvas = FigureCanvasTkAgg(fig_hist, master=frame)
    canvas.get_tk_widget().pack(side="left", fill='both', expand=True)
    canvas.draw()


def load_hist_plots_volt(root=None, df_slice_sci=None, start_time=None, end_time=None, bins=None,
                         cmin=None, cmax=None, density=None, norm=None, unit=None, channel1=None,
                         channel2=None, row=None, column=None, sticky=None, columnspan=None,
                         rowspan=None, fig_width=None, fig_height=None, v_min=2.2, v_max=3.9,
                         cmap=None):
    """
    Loads the histogram plots for the selected time range and displays them in the GUI. This is for
    the voltage

    Parameters
    ----------
    df_slice_sci : pandas.DataFrame
        The dataframe containing the science data.
    start_time : str
        The start time of the time range to be plotted.
    end_time : str
        The end time of the time range to be plotted.
    channel1 : str
        The channel to be plotted.
    channel2 : str
        The channel to be plotted.
    row : int
        The row in which the plots should be displayed.
    column : int
        The column in which the plots should be displayed.
    sticky : str
        The sticky parameter for the grid.
    columnspan : int
        The number of columns the plots should span.
    rowspan : int
        The number of rows the plots should span.
    fig_width : float
        The width of the figure.
    fig_height : float
        The height of the figure.

    Returns
    -------
    None
    """
    # Set the fontstyle to Times New Roman
    font_mpl = {'family': 'serif', 'weight': 'normal'}
    plt.rc('font', **font_mpl)
    plt.rc('text', usetex=False)

    fig_hist = lgpr.plot_data_class(
        df_slice_sci=df_slice_sci, start_time=start_time, end_time=end_time, bins=bins, cmin=cmin,
        cmax=cmax, density=density, norm=norm, unit=unit, channel1=channel1, channel2=channel2,
        volt_fig_width=fig_width, volt_fig_height=fig_height, v_min=v_min, v_max=v_max, cmap=cmap
    ).hist_plots_volt()
    fig_hist.tight_layout()
    frame = tk.Frame(root)
    frame.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky=sticky)
    canvas = FigureCanvasTkAgg(fig_hist, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side='left', fill='both', expand=True)


def load_all_hist_plots(
        root=None, df_slice_sci=None, start_time=None, end_time=None, bins=None, cmin=None,
        cmax=None, x_min=None, x_max=None, y_min=None, y_max=None, density=None, norm=None,
        unit=None, row_hist=3, col_hist=1, channel1=None, channel3=None, row_channel13=None,
        column_channel13=None, sticky_channel13=None, row_span_channel13=None,
        column_span_channel13=None, channel2=None, channel4=None, row_channel24=None,
        column_channel24=None, sticky_channel24=None, row_span_channel24=None,
        column_span_channel24=None, hist_fig_height=None, hist_fig_width=None, hist_colspan=None,
        hist_rowspan=None, channel13_fig_height=None, channel13_fig_width=None,
        channel24_fig_height=None, channel24_fig_width=None, v_min=None, v_max=None,
        v_sum_min=None, v_sum_max=None, cut_status_var=None, crv_fit=None, lin_corr=None, cmap=None,
        use_fig_size=False
):
    """
    Loads the histogram plots for the selected time range and displays them in the GUI. This is for
    the voltage as well as the x,y locations of the photons

    Parameters
    ----------
    df_slice_sci : pandas.DataFrame
        The dataframe containing the science data.
    start_time : str
        The start time of the time range to be plotted.
    end_time : str
        The end time of the time range to be plotted.
    bins : int
        The number of bins to be used for the histogram.
    cmin : int
        The minimum value of the colorbar.
    cmax : int
        The maximum value of the colorbar.
    x_min : int
        The minimum value of the x-axis.
    x_max : int
        The maximum value of the x-axis.
    y_min : int
        The minimum value of the y-axis.
    y_max : int
        The maximum value of the y-axis.
    density : bool
        Whether or not the histogram should be normalized.
    norm : bool
        Whether or not the histogram should be normalized.
    unit : str
        The unit of the axes.
    row_hist : int
        The row in which the histogram plots should be displayed.
    channel1 : str
        The channel to be plotted.
    channel3 : str
        The channel to be plotted.
    row_channel13 : int
        The row in which the histogram plots should be displayed.
    column_channel13 : int
        The column in which the histogram plots should be displayed.
    sticky_channel13 : str
        The sticky parameter for the grid.
    channel2 : str
        The channel to be plotted.
    channel4 : str
        The channel to be plotted.
    row_channel24 : int
        The row in which the histogram plots should be displayed.
    column_channel24 : int
        The column in which the histogram plots should be displayed.
    sticky_channel24 : str
        The sticky parameter for the grid.
    hist_fig_height : float
        The height of the main histogram figure.
    hist_fig_width : float
        The width of the main histogram figure.
    hist_colspan : int
        The number of columns the histogram plots should span.
    hist_rowspan : int
        The number of rows the histogram plots should span.
    channel13_fig_height : float
        The height of the histogram figure for the channel 1 and 3.
    channel13_fig_width : float
        The width of the histogram figure for the channel 1 and 3.
    channel24_fig_height : float
        The height of the histogram figure for the channel 2 and 4.
    channel24_fig_width : float
        The width of the histogram figure for the channel 2 and 4.

    Returns
    -------
        None
    """
    load_hist_plots(root=root[0], df_slice_sci=df_slice_sci, start_time=start_time,
                    end_time=end_time, bins=bins, cmin=cmin, cmax=cmax, x_min=x_min, x_max=x_max,
                    y_min=y_min, y_max=y_max, density=density, norm=norm, unit=unit, row=row_hist,
                    column=col_hist, fig_height=hist_fig_height, fig_width=hist_fig_width,
                    columnspan=hist_colspan, rowspan=hist_rowspan, v_min=v_min, v_max=v_max,
                    v_sum_min=v_sum_min, v_sum_max=v_sum_max, cut_status_var=cut_status_var,
                    crv_fit=crv_fit, lin_corr=lin_corr, cmap=cmap, use_fig_size=use_fig_size)

    load_hist_plots_volt(root=root[1], df_slice_sci=df_slice_sci, start_time=start_time,
                         end_time=end_time, bins=bins, cmin=cmin, cmax=cmax, density=density,
                         norm=norm, channel1=channel1, channel2=channel3,
                         row=row_channel13, column=column_channel13, sticky=sticky_channel13,
                         rowspan=row_span_channel13, columnspan=column_span_channel13,
                         fig_width=channel13_fig_width, fig_height=channel13_fig_height,
                         v_min=v_min, v_max=v_max, cmap=cmap)

    load_hist_plots_volt(root=root[1], df_slice_sci=df_slice_sci, start_time=start_time,
                         end_time=end_time, bins=bins, cmin=cmin, cmax=cmax, density=density,
                         norm=norm, channel1=channel2, channel2=channel4, row=row_channel24,
                         column=column_channel24, sticky=sticky_channel24,
                         rowspan=row_span_channel24, columnspan=column_span_channel24,
                         fig_width=channel24_fig_width, fig_height=channel24_fig_height,
                         v_min=v_min, v_max=v_max, cmap=cmap)
