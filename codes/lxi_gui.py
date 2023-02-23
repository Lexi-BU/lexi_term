import importlib
import platform
import tkinter as tk
from tkinter import font, ttk

import global_variables
import lxi_gui_entry_box as lgeb
import lxi_gui_plot_routines as lgpr
import lxi_load_plot_routines as llpr
import lxi_misc_codes as lmsc
import lxi_file_read_funcs as lxrf
# import lxi_csv_to_cdf as lctc

importlib.reload(lgpr)
importlib.reload(lxrf)
importlib.reload(global_variables)
importlib.reload(llpr)
importlib.reload(lgeb)
importlib.reload(lmsc)
# importlib.reload(lctc)

# Initialize the global variables. This is necessary because the global variables is where all the
# data and name of the files are stored.
global_variables.init()


#def save_cdf():
#    """
#    The function, upon clicking the "Save CDF" button, saves the data in the csv file to a cdf file.
#    """
#    try:
#        inputs = {
#            "df": global_variables.all_file_details["df_all_sci"],
#            "csv_file": global_variables.all_file_details["file_name_sci"],
#        }
#
#        lctc.lxi_csv_to_cdf(**inputs)
#    except Exception as e:
#        print(f"\n \x1b[1;31;255m Error: \x1b[0m Could not save the cdf file. Following exception"
#              f" was raised: \n \x1b[1;31;255m {e} \x1b[0m is not defined. \n Check if a valid "
#              f"Science csv file is loaded. \n")
#        pass
#

def hist_plot_inputs(dpi=100):
    """
    The function creates and updates the list of widget inputs as might be available from the GUI
    and plots all the histograms.
    """

    inputs = {
        "root": [sci_tab, sci_tab],
        "df_slice_sci": global_variables.all_file_details["df_slice_sci"],
        "start_time": start_time.get(),
        "end_time": end_time.get(),
        "bins": hist_bins_entry.get(),
        "cmin": c_min_entry.get(),
        "cmax": c_max_entry.get(),
        "x_min": x_min_entry.get(),
        "x_max": x_max_entry.get(),
        "y_min": y_min_entry.get(),
        "y_max": y_max_entry.get(),
        "density": density_status_var.get(),
        "norm": norm_type_var.get(),
        "row_hist": 0,
        "col_hist": 2,
        "channel1": "Channel1",
        "channel2": "Channel2",
        "row_channel13": 0,
        "column_channel13": 7,
        "sticky_channel13": "nesw",
        "row_span_channel13": 4,
        "column_span_channel13": 4,
        "channel3": "Channel3",
        "channel4": "Channel4",
        "row_channel24": 7,
        "column_channel24": 7,
        "sticky_channel24": "nesw",
        "row_span_channel24": 4,
        "column_span_channel24": 4,
        "hist_fig_height": screen_height / (2 * dpi),
        "hist_fig_width": screen_width / (2 * dpi),
        "hist_colspan": 2,
        "hist_rowspan": 20,
        "channel13_fig_height": screen_height / (3 * dpi),
        "channel13_fig_width": screen_width / (3 * dpi),
        "channel24_fig_height": screen_height / (3 * dpi),
        "channel24_fig_width": screen_width / (3 * dpi),
        "v_min": v_min_thresh_entry.get(),
        "v_max": v_max_thresh_entry.get(),
        "v_sum_min": v_sum_min_thresh_entry.get(),
        "v_sum_max": v_sum_max_thresh_entry.get(),
        "crv_fit": curve_fit_status_var.get(),
        "use_fig_size": True
    }

    llpr.load_all_hist_plots(**inputs)


def ts_plot_inputs(plot_opt_entry=None, dpi=100, row=None, column=None, columnspan=3, rowspan=2,
                   plot_key=None):
    """
    The function creates and updates the list of widget inputs as might be available from the GUI
    and plots time series, one at a time.
    """
    inputs = {
        "root": hk_tab,
        "df_slice_hk": global_variables.all_file_details["df_slice_hk"],
        "plot_key": plot_opt_entry.get(),
        "start_time": start_time.get(),
        "end_time": end_time.get(),
        "row": row,
        "column": column,
        "columnspan": columnspan,
        "rowspan": rowspan,
        "fig_width": screen_width / (4 * dpi),
        "fig_height": screen_height / (6 * dpi)
    }

    llpr.load_ts_plots(**inputs)


def ts_button_val_change():
    """
    This function is called when the "Default Options" button is clicked. It sets the values of all
    the 9 time series plot options to the default values.
    """

    default_key_list = ['PinPullerTemp', 'OpticsTemp', 'LEXIbaseTemp',
                        '+5.2V_Imon', '+10V_Imon', '+3.3V_Imon',
                        '+28V_Imon', 'DeltaEvntCount', 'DeltaDroppedCount']
    plot_opt_entry_list = [plot_opt_entry_1, plot_opt_entry_2, plot_opt_entry_3,
                           plot_opt_entry_4, plot_opt_entry_5, plot_opt_entry_6,
                           plot_opt_entry_7, plot_opt_entry_8, plot_opt_entry_9]

    if default_opt_var.get() == True:
        for i in range(len(default_key_list)):
            plot_opt_entry_list[i].set(default_key_list[i])


def refresh_ts_plot():
    """
    Refresh the time series plot
    """
    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_1, row=1, column=0, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_2, row=1, column=3, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_3, row=1, column=6, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_4, row=3, column=0, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_5, row=3, column=3, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_6, row=3, column=6, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_7, row=5, column=0, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_8, row=5, column=3, rowspan=1, columnspan=3)
    except Exception:
        pass

    try:
        ts_plot_inputs(plot_opt_entry=plot_opt_entry_9, row=5, column=6, rowspan=1, columnspan=3)
    except Exception:
        pass


# Create the main window.
root = tk.Tk()

# Get the DPI of the screen. This is used to scale the figure size.
dpi = root.winfo_fpixels('1i')

# NOTE: This hack is necessary since I am using multiple monitors. This can be edited as we work on
# a different machine.
# Check whether the operating system is windows or linux, and assign the correct screen width and
# height.
if platform.system() == "Windows":
    screen_width, screen_height = 0.9 * root.winfo_screenwidth(), 0.9 * root.winfo_screenheight()
if platform.system() == "Linux":
    screen_width, screen_height = 0.45 * root.winfo_screenwidth(), 0.8 * root.winfo_screenheight()
else:
    screen_width, screen_height = 0.9 * root.winfo_screenwidth(), 0.9 * root.winfo_screenheight()

screen_width = 2000
screen_height = 1000
print("If the GUI size is messed up, check comment on line #201 of the code 'lxi_gui.py'.")

# Set the title of the main window.
root.title("LEXI GUI")
# Add the lxi logo
# NOTE: This doesn't work on UNIX system. Couldn't find a solution.
# root.iconbitmap("../figures/lxi_icon.ico")
# set size of you window here is example for screen height and width
root.geometry(f"{int(screen_width * 0.9)}x{int(screen_height * 0.9)}")

# if the window is resized, the figure will be scaled accordingly
root.resizable(width=True, height=True)
# redefine screen_width and screen_height if the window is resized.
root.bind("<Configure>", lambda event: root.update_idletasks())

# Create two tabs corresponding to science and housekeeping stuff.
tabControl = ttk.Notebook(root)
tabControl.pack(expand=1, fill="both")
tabControl.pack(expand=1, fill="both")
sci_tab = tk.Frame(tabControl)
sci_tab.pack(expand=1, fill="both")
hk_tab = tk.Frame(tabControl)
hk_tab.pack(expand=1, fill="both")

tabControl.add(sci_tab, text='Science Stuff')
tabControl.add(hk_tab, text='Housekeeping Stuff')

# Configure the science tab rows and columns.
sci_tab.columnconfigure(0, {'minsize': 1}, weight=1)
sci_tab.columnconfigure(1, {'minsize': 1}, weight=2)
sci_tab.columnconfigure(2, {'minsize': 1}, weight=5)
sci_tab.columnconfigure(3, {'minsize': 1}, weight=5)
sci_tab.columnconfigure(4, {'minsize': 1}, weight=1)
sci_tab.columnconfigure(5, {'minsize': 1}, weight=1)
sci_tab.columnconfigure(6, {'minsize': 1}, weight=1)
sci_tab.columnconfigure(7, {'minsize': 1}, weight=1)
sci_tab.columnconfigure(8, {'minsize': 1}, weight=1)
sci_tab.columnconfigure(9, {'minsize': 1}, weight=1)

# Configure the sci_tab rows
# for i in range(0, 20):
#     sci_tab.rowconfigure(i, {'minsize': 0}, weight=0)

sci_tab.configure(bg="white", padx=5, pady=5, relief="raised", borderwidth=5, highlightthickness=5)
hk_tab.configure(padx=5, pady=5, relief="raised", borderwidth=5, highlightthickness=5)

# Configure the housekeeping tab rows and columns.
for i in range(0, 10):
    hk_tab.rowconfigure(i, {'minsize': 1}, weight=1)
    hk_tab.columnconfigure(i, {'minsize': 1}, weight=1)

# Choose a font style for GUI
font_style = font.Font(family="serif", size=12)
font_style_box = font.Font(family="serif", size=12, weight="bold")
font_style_big = font.Font(family="serif", size=25)

# Insert a file load button
# For science file
sci_file_load_button = tk.Button(sci_tab, text="Load Science File", command=lxrf.open_file_sci,
                                 font=font_style)
sci_file_load_button.grid(row=0, column=0, columnspan=1, pady=0, sticky="ew")

sci_file_name = tk.StringVar()
sci_file_name.set("No file loaded")
sci_file_load_entry = tk.Entry(sci_tab, textvariable=sci_file_name, font=font_style,
                               justify="left", bg="snow", fg="black", relief="sunken",
                               borderwidth=2)
sci_file_load_entry.grid(row=1, column=0, columnspan=2, pady=0, sticky="ew")

# insert the file_load_entry value into the entry box only if the sci_file_load_button is clicked
sci_file_load_button.config(
    command=lambda: sci_file_load_entry.insert(0, lxrf.open_file_sci()))

# For housekeeping file
hk_file_load_button = tk.Button(sci_tab, text="Load HK File", command=lxrf.open_file_hk,
                                font=font_style)
hk_file_load_button.grid(row=2, column=0, columnspan=1, pady=0, sticky="ew")

hk_file_name = tk.StringVar()
hk_file_name.set("No file loaded")
hk_file_load_entry = tk.Entry(sci_tab, textvariable=hk_file_name, font=font_style, justify="left",
                              bg="snow", fg="black", relief="sunken", borderwidth=2)
hk_file_load_entry.grid(row=3, column=0, columnspan=2, pady=0, sticky="ew")

# insert the file_load_entry value into the entry box only if the hk_file_load_button is clicked
hk_file_load_button.config(
    command=lambda: hk_file_load_entry.insert(0, lxrf.open_file_hk()))

# For binary file
b_file_load_button = tk.Button(sci_tab, text="Load binary File", command=lxrf.open_file_b,
                               font=font_style)
b_file_load_button.grid(row=4, column=0, columnspan=1, pady=0, sticky="ew")

b_file_name = tk.StringVar()
b_file_name.set("No file loaded")
b_file_load_entry = tk.Entry(sci_tab, textvariable=b_file_name, font=font_style, justify="left",
                             bg="snow", fg="black", relief="sunken", borderwidth=2)
b_file_load_entry.grid(row=5, column=0, columnspan=2, pady=0, sticky="ew")

# insert the file_load_entry value into the entry box only if the b_file_load_button is clicked
b_file_load_button.config(command=lambda: b_file_load_entry.insert(0, lxrf.open_file_b()))

# If a new file is loaded, then print its name in the entry box and update the file_name variable.
sci_file_name.trace("w", lambda *_: sci_file_name.set(lmsc.file_name_update(file_type="sci")))
hk_file_name.trace("w", lambda *_: hk_file_name.set(lmsc.file_name_update(file_type="hk")))

# If a new binary file is loaded, then update the name of all three files.
b_file_name.trace("w", lambda *_: b_file_name.set(lmsc.file_name_update(file_type="b")))
b_file_name.trace("w", lambda *_: sci_file_name.set(lmsc.file_name_update(file_type="sci")))
b_file_name.trace("w", lambda *_: hk_file_name.set(lmsc.file_name_update(file_type="hk")))

# If the global_variables.all_file_details["df_slice_hk"] is not empty, then set the comlumn names
# to the columns in the dataframe
if bool("df_slice_hk" in global_variables.all_file_details.keys()):
    ts_options = global_variables.all_file_details["df_slice_hk"].columns.tolist()
else:
    ts_options = ['HK_id', 'PinPullerTemp', 'OpticsTemp', 'LEXIbaseTemp', 'HVsupplyTemp',
                  '+5.2V_Imon', '+10V_Imon', '+3.3V_Imon', 'AnodeVoltMon', '+28V_Imon',
                  'ADC_Ground', 'Cmd_count', 'Pinpuller_Armed', 'HVmcpAuto', 'HVmcpMan',
                  'DeltaEvntCount', 'DeltaDroppedCount', 'DeltaLostEvntCount']

# Plot options for the first plot
plot_opt_label_1 = tk.Label(hk_tab, text="Plot options:", font=font_style_box)
plot_opt_label_1.grid(row=0, column=0, columnspan=1, pady=0, sticky="w")
plot_opt_entry_1 = tk.StringVar(hk_tab)
plot_opt_entry_1.set("Select a column")
ts_menu_1 = tk.OptionMenu(hk_tab, plot_opt_entry_1, *ts_options)
ts_menu_1.grid(row=0, column=2, columnspan=1, sticky="w")

# Plot options for the second plot
plot_opt_entry_2 = tk.StringVar(hk_tab)
plot_opt_entry_2.set("Select a column")
ts_menu_2 = tk.OptionMenu(hk_tab, plot_opt_entry_2, *ts_options)
ts_menu_2.grid(row=0, column=5, columnspan=1, sticky="w")

# Plot optiosn for third plot
plot_opt_entry_3 = tk.StringVar(hk_tab)
plot_opt_entry_3.set("Select a column")
ts_menu_3 = tk.OptionMenu(hk_tab, plot_opt_entry_3, *ts_options)
ts_menu_3.grid(row=0, column=8, columnspan=1, sticky="w")

# Plot options for fourth plot (in the second row)
plot_opt_entry_4 = tk.StringVar(hk_tab)
plot_opt_entry_4.set("Select a column")
ts_menu_4 = tk.OptionMenu(hk_tab, plot_opt_entry_4, *ts_options)
ts_menu_4.grid(row=2, column=2, columnspan=1, sticky="w")

# Plot options for fifth plot (in the second row)
plot_opt_entry_5 = tk.StringVar(hk_tab)
plot_opt_entry_5.set("Select a column")
ts_menu_5 = tk.OptionMenu(hk_tab, plot_opt_entry_5, *ts_options)
ts_menu_5.grid(row=2, column=5, columnspan=1, sticky="w")

# Plot options for sixth plot (in the second row)
plot_opt_entry_6 = tk.StringVar(hk_tab)
plot_opt_entry_6.set("Select a column")
ts_menu_6 = tk.OptionMenu(hk_tab, plot_opt_entry_6, *ts_options)
ts_menu_6.grid(row=2, column=8, columnspan=1, sticky="w")

# Plot options for seventh plot (in the third row)
plot_opt_entry_7 = tk.StringVar(hk_tab)
plot_opt_entry_7.set("Select a column")
ts_menu_7 = tk.OptionMenu(hk_tab, plot_opt_entry_7, *ts_options)
ts_menu_7.grid(row=4, column=2, columnspan=1, sticky="w")

# Plot options for eighth plot (in the third row)
plot_opt_entry_8 = tk.StringVar(hk_tab)
plot_opt_entry_8.set("Select a column")
ts_menu_8 = tk.OptionMenu(hk_tab, plot_opt_entry_8, *ts_options)
ts_menu_8.grid(row=4, column=5, columnspan=1, sticky="w")

# Plot options for ninth plot (in the third row)
plot_opt_entry_9 = tk.StringVar(hk_tab)
plot_opt_entry_9.set("Select a column")
ts_menu_9 = tk.OptionMenu(hk_tab, plot_opt_entry_9, *ts_options)
ts_menu_9.grid(row=4, column=8, columnspan=1, sticky="w")

# The minimum value of x-axis for histogram plot
x_min_entry = lgeb.entry_box(root=sci_tab, row=0, column=4, entry_label="X-min", entry_val=0.35,
                             font_style=font_style_box)

# The maximum value of x-axis for histogram plot
x_max_entry = lgeb.entry_box(root=sci_tab, row=1, column=4, entry_label="X-max", entry_val=0.62,
                             font_style=font_style_box)

# The minimum value of y-axis for histogram plot
y_min_entry = lgeb.entry_box(root=sci_tab, row=2, column=4, entry_label="Y-min", entry_val=0.35,
                             font_style=font_style_box)

# The maximum value of y-axis for histogram plot
y_max_entry = lgeb.entry_box(root=sci_tab, row=3, column=4, entry_label="Y-max", entry_val=0.62,
                             font_style=font_style_box)

# The number of bins for histogram plot
hist_bins_entry = lgeb.entry_box(root=sci_tab, row=4, column=4, entry_label="Bins", entry_val=100,
                                 font_style=font_style_box)

# Mimimum number of data points in each bin for the histogram plot
c_min_entry = lgeb.entry_box(root=sci_tab, row=5, column=4, entry_label="C Min", entry_val=1,
                             font_style=font_style_box)

# Maximum number of data points in each bin for the histogram plot
c_max_entry = lgeb.entry_box(root=sci_tab, row=6, column=4, entry_label="C Max", entry_val="None",
                             font_style=font_style_box)

# Choose whether to plot probability density or the number of data points in each bin (is Bool)
density_label = tk.Label(sci_tab, text="Density", font=font_style_box, bg="white")
density_label.grid(row=7, column=5, columnspan=1, sticky="n")

# Add a checkbox to choose whether to plot probability density or the number of data points in each
# bin
density_status_var = tk.BooleanVar()
density_status_var.set(False)
density_checkbox = tk.Checkbutton(sci_tab, text="", font=font_style_box,
                                  variable=density_status_var, bg="white", fg="black")
density_checkbox.grid(row=7, column=4, columnspan=1, sticky="n")

# Redo the histogram plot if the status of the checkbox is changed
density_status_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

# Key for the norm of the colorbar
norm_label = tk.Label(sci_tab, text="Norm", font=font_style_box, bg="white", fg="black")
norm_label.grid(row=8, column=5, columnspan=1, sticky="n")

# Add radio button for the norm type (default is 'log', other option is 'linear')
norm_type_var = tk.StringVar()
norm_type_var.set("log")
norm_type_1 = tk.Radiobutton(sci_tab, text="Log", variable=norm_type_var, value="log", bg="white",
                             fg="black")
norm_type_1.grid(row=8, column=4, columnspan=1, sticky="new")

norm_type_2 = tk.Radiobutton(sci_tab, text="Lin", variable=norm_type_var, value="linear",
                             bg="white", fg="black")
norm_type_2.grid(row=9, column=4, columnspan=1, sticky="new")

# Redo the histogram plot when the norm type is changed
norm_type_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

# Minimum threshold for the voltage to be considered
v_min_thresh_entry = lgeb.entry_box(root=sci_tab, row=10, column=4, entry_label="V Min",
                                    entry_val=1.2, font_style=font_style_box)

# Maximum threshold for the voltage to be considered
v_max_thresh_entry = lgeb.entry_box(root=sci_tab, row=11, column=4, entry_label="V Max",
                                    entry_val=4, font_style=font_style_box)

# Sum of minimum and maximum threshold for the voltage to be considered
v_sum_min_thresh_entry = lgeb.entry_box(root=sci_tab, row=12, column=4, entry_label="V sum Min",
                                        entry_val=3, font_style=font_style_box)
v_sum_max_thresh_entry = lgeb.entry_box(root=sci_tab, row=13, column=4, entry_label="V sum Max",
                                        entry_val=8, font_style=font_style_box)

# Choose whether to plot curve fit or not (is Bool)
curve_fit_label = tk.Label(sci_tab, text="Curve Fit", font=font_style_box, bg="white", fg="black")
curve_fit_label.grid(row=14, column=5, columnspan=1, sticky="n")

# Add a checkbox to choose whether to plot curve fit or not
curve_fit_status_var = tk.BooleanVar()
curve_fit_status_var.set(False)
curve_fit_checkbox = tk.Checkbutton(sci_tab, text="", font=font_style_box,
                                    variable=curve_fit_status_var, bg="white", fg="black")
curve_fit_checkbox.grid(row=14, column=4, columnspan=1, sticky="n")

curve_fit_status_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

# Add a button to save the data to a cdf file
#cdf_save_button = tk.Button(
#    sci_tab, text="Save CDF", command=lambda: save_cdf(), font=font_style_box,
#    justify="center", bg="snow", fg="green", pady=5, padx=5, borderwidth=2,
#    relief="raised", highlightthickness=2, highlightbackground="green", highlightcolor="green"
#)
#cdf_save_button.grid(row=12, column=7, columnspan=2, sticky="n")

# Label for plot times
start_time_label = tk.Label(sci_tab, text="Plot Times", font=font_style, bg="white", fg="black")
start_time_label.grid(row=6, column=0, columnspan=2, sticky="nsew")

# Add an input box with a label for start time
start_time = tk.Entry(sci_tab, justify="center", bg="snow", fg="green", borderwidth=2)
start_time.insert(0, "YYYY-MM-DD HH:MM:SS")
start_time.grid(row=7, column=0, columnspan=2, sticky="nsew")
start_time_label = tk.Label(sci_tab, text="Start Time", font=font_style, bg="white", fg="black")
start_time_label.grid(row=8, column=0, columnspan=2, sticky="nsew")

# Add an input box with a label for end time
# end_time_entry, end_time_label = lgeb.entry_box(root=sci_tab, row=17, column=3,
#                                                 entry_label="End Time", width=30,
#                                                 entry_val="YYYY-MM-DD HH:MM:SS",
#                                                 font_style=font_style)
end_time = tk.Entry(sci_tab, justify="center", bg="snow", fg="green", borderwidth=2)
end_time.insert(0, "YYYY-MM-DD HH:MM:SS")
end_time.grid(row=9, column=0, columnspan=2, sticky="nsew")
end_time_label = tk.Label(sci_tab, text="End Time", font=font_style, bg="white", fg="black")
end_time_label.grid(row=10, column=0, columnspan=2)

# if any of the ts_options are changed, update the plot
plot_opt_entry_1.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_1, row=1, column=0, rowspan=1,
                                   columnspan=3))

plot_opt_entry_2.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_2, row=1, column=3, rowspan=1,
                                   columnspan=3))

plot_opt_entry_3.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_3, row=1, column=6, rowspan=1,
                                   columnspan=3))

plot_opt_entry_4.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_4, row=3, column=0, rowspan=1,
                                   columnspan=3))

plot_opt_entry_5.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_5, row=3, column=3, rowspan=1,
                                   columnspan=3))

plot_opt_entry_6.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_6, row=3, column=6, rowspan=1,
                                   columnspan=3))

plot_opt_entry_7.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_7, row=5, column=0, rowspan=1,
                                   columnspan=3))

plot_opt_entry_8.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_8, row=5, column=3, rowspan=1,
                                   columnspan=3))

plot_opt_entry_9.trace(
    "w", lambda *_: ts_plot_inputs(plot_opt_entry=plot_opt_entry_9, row=5, column=6, rowspan=1,
                                   columnspan=3))

# If the plot button is pressed then all the histogram plots are redrawn
plot_button = tk.Button(sci_tab, text="Plot Histogram", font=font_style_box, justify="center",
                        command=lambda: hist_plot_inputs(dpi=dpi))

plot_button.grid(row=11, column=0, columnspan=1, rowspan=1, sticky="nsew", pady=5, padx=5)

# If the plot button is pressed, then print the current time
plot_button.bind("<Button-1>", lambda event: lmsc.print_time_details(start_time=start_time.get(),
                                                                     end_time=end_time.get()))

# Add a quit button
quit_button_sci = tk.Button(
    sci_tab, text="Quit", command=root.destroy, font=font_style_box, justify="center", bg="snow",
    fg="red", pady=5, padx=5, borderwidth=2, relief="raised", highlightthickness=2,
    highlightbackground="red", highlightcolor="red"
)
quit_button_sci.grid(row=12, column=0, columnspan=1, rowspan=1, sticky="n")

# Add a default option check box
default_opt_var = tk.BooleanVar()
default_opt_var.set(False)
default_opt_checkbox = tk.Checkbutton(hk_tab, text="Default Options", font=font_style_box,
                                      variable=default_opt_var, bg="white", fg="black")
default_opt_checkbox.grid(row=12, column=1, columnspan=1, sticky="n")

default_opt_var.trace("w", lambda *_: ts_button_val_change())

# If Default Checkbox is checked, then set the default options for the time series data

# Add a refresh button to reload all the time series plots
refresh_ts_hk_button = tk.Button(
    hk_tab, text="Refresh", command=lambda: refresh_ts_plot(), font=font_style_box,
    justify="center", bg="snow", fg="green", pady=5, padx=5, borderwidth=2,
    relief="raised", highlightthickness=2, highlightbackground="green", highlightcolor="green"
)
refresh_ts_hk_button.grid(row=12, column=2, columnspan=2, rowspan=1, sticky="n")

quit_button_hk = tk.Button(
    hk_tab, text="Quit", command=root.destroy, font=font_style_box, justify="center", bg="snow",
    fg="red", pady=5, padx=5, borderwidth=2, relief="raised", highlightthickness=2,
    highlightbackground="red", highlightcolor="red"
)
quit_button_hk.grid(row=12, column=4, columnspan=2, rowspan=1, sticky="n")

# blank_label = tk.Label(sci_tab, text="", font=font_style_box, bg="white")
# for row in range(10):
#     blank_label.grid(row=11 + row, column=0, columnspan=2, sticky="nsew")
#     blank_label.grid(row=12 + row, column=4, columnspan=5, sticky="nsew")

# blank_label = tk.Label(sci_tab, text="", font=font_style_box, bg="white")
# for row in range(10):
#     blank_label.grid(row=11+row, column=0, columnspan=2, sticky="nsew")
#     blank_label.grid(row=12+row, column=4, columnspan=5, sticky="nsew")

root.mainloop()
