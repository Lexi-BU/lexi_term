import importlib
import logging
import os
import platform
import tkinter as tk
from tkinter import font, ttk

import global_variables
import lxi_file_read_funcs as lxrf
import lxi_gui_config as lgcf
import lxi_gui_entry_box as lgeb
import lxi_gui_plot_routines as lgpr
import lxi_load_plot_routines as llpr
import lxi_misc_codes as lmsc

importlib.reload(lgpr)
importlib.reload(lxrf)
importlib.reload(global_variables)
importlib.reload(llpr)
importlib.reload(lgeb)
importlib.reload(lmsc)
importlib.reload(lgcf)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler("../log/lxi_gui.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Initialize the global variables. This is necessary because the global variables is where all the
# data and name of the files are stored.
global_variables.init()


def hist_plot_inputs(dpi=100):
    """
    The function creates and updates the list of widget inputs as might be available from the GUI
    and plots all the histograms.
    """

    if global_variables.all_file_details:
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
            "unit": unit_type_var.get(),
            "row_hist": 0,
            "col_hist": 2,
            "channel1": "Channel1",
            "channel2": "Channel2",
            "row_channel13": 0,
            "column_channel13": 11,
            "sticky_channel13": "nesw",
            "row_span_channel13": 6,
            "column_span_channel13": 3,
            "channel3": "Channel3",
            "channel4": "Channel4",
            "row_channel24": 8,
            "column_channel24": 11,
            "sticky_channel24": "nesw",
            "row_span_channel24": 6,
            "column_span_channel24": 3,
            "hist_fig_height": screen_height / (2.2 * dpi),
            "hist_fig_width": screen_width / (2.2 * dpi),
            "hist_colspan": 7,
            "hist_rowspan": 20,
            "channel13_fig_height": screen_height / (3 * dpi),
            "channel13_fig_width": screen_width / (3 * dpi),
            "channel24_fig_height": screen_height / (3 * dpi),
            "channel24_fig_width": screen_width / (3 * dpi),
            "v_min": v_min_thresh_entry.get(),
            "v_max": v_max_thresh_entry.get(),
            "v_sum_min": v_sum_min_thresh_entry.get(),
            "v_sum_max": v_sum_max_thresh_entry.get(),
            "cut_status_var": cut_status_var.get(),
            "crv_fit": curve_fit_status_var.get(),
            "lin_corr": lin_corr_status_var.get(),
            "cmap": cmap_option.get(),
            "use_fig_size": True,
            "dark_mode": dark_mode_var.get(),
        }

        llpr.load_all_hist_plots(**inputs)
    else:
        logger.info("No data to plot")


def ts_plot_inputs(
    plot_opt_entry=None,
    dpi=100,
    row=None,
    column=None,
    columnspan=3,
    rowspan=2,
    plot_key=None,
):
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
        "fig_width": screen_width / (2 * dpi),
        "fig_height": screen_height / (3 * dpi),
        "dark_mode": dark_mode_var.get(),
        "time_type": time_type.get(),
    }
    llpr.load_ts_plots(**inputs)


def ts_button_val_change(default_opt_var):
    """
    This function is called when the "Default Options" button is clicked. It sets the values of all
    the 9 time series plot options to the default values.
    """

    default_key_list = [
        # "PinPullerTemp",
        # "OpticsTemp",
        "HVsupplyTemp",
        "LEXIbaseTemp",
        "+3.3V_Imon",
        "+5.2V_Imon",
        "+10V_Imon",
        "+28V_Imon",
        "AnodeVoltMon",
        "DeltaEvntCount",
        # "DeltaLostEvntCount",
        "DeltaDroppedCount",
    ]
    plot_opt_entry_list = [
        plot_opt_entry_1,
        plot_opt_entry_2,
        plot_opt_entry_3,
        plot_opt_entry_4,
        plot_opt_entry_5,
        plot_opt_entry_6,
        plot_opt_entry_7,
        plot_opt_entry_8,
        plot_opt_entry_9,
    ]

    # Check if global_variables.all_file_details["df_slice_hk"] is empty, if it is then return from
    # the function without doing anything else refresh the time series plot
    if global_variables.all_file_details:
        if default_opt_var.get() is True:
            for i in range(len(default_key_list)):
                plot_opt_entry_list[i].set(default_key_list[i])
    else:
        logger.info("No time series data to plot")


def refresh_ts_plot():
    """
    Refresh the time series plot
    """
    if global_variables.all_file_details:
        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_1, row=1, column=0, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_1.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_2, row=1, column=3, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_2.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_3, row=1, column=6, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_3.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_4, row=3, column=0, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_4.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_5, row=3, column=3, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_5.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_6, row=3, column=6, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_6.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_7, row=5, column=0, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_7.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_8, row=5, column=3, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_8.get()}"
            )
            pass

        try:
            ts_plot_inputs(
                plot_opt_entry=plot_opt_entry_9, row=5, column=6, rowspan=1, columnspan=3
            )
        except Exception:
            logger.exception(
                "Exception occurred while refreshing the time series plot for"
                f"{plot_opt_entry_9.get()}"
            )
            pass
    else:
        logger.info("No time series data to plot")


def load_and_copy_files():
    """
    This function is called when the "Load Files" button is clicked. It copies the LEXI data files
    from PIT to the local computer and loads the data.
    """
    # try:
    #     lmsc.copy_pit_files()
    # except Exception as e:
    #     logger.exception(f"Exception occurred while copying files from PIT: {e}")
    #     pass

    lmsc.load_folder(
        file_val=folder_path.get(),
        t_start=start_time.get(),
        t_end=end_time.get(),
        multiple_files=multi_file_status_var.get(),
    )


    # Try to save the csv file
    try:
        lmsc.save_csv()
    except Exception as e:
        logger.exception(f"Exception occurred while saving the csv file: {e}")
        pass

    # Try to save the cdf file
    try:
        lmsc.save_cdf()
    except Exception as e:
        logger.exception(f"Exception occurred while saving the cdf file: {e}")
        pass

    # try:
    #     default_opt_var.set(True)
    #     ts_button_val_change(default_opt_var)
    #     refresh_ts_plot()
    # except Exception as e:
    #     logger.exception(f"Exception occurred while refreshing the time series plot: {e}")
    #     pass

    # try:
    #     hist_plot_inputs()
    # except Exception as e:
    #     logger.exception(f"Exception occurred while refreshing the histogram plot: {e}")
    #     pass


def update_time_entry(time_entry, time_entry_other):
    """
    This function will take one time entry and update the other
    """
    time_entry_other.delete(0, tk.END)
    time_entry_other.insert(0, time_entry.get())


def dark_mode_change():
    """
    This function is called when the "Dark Mode" checkbox is clicked. It changes the background and
    foreground colors of the GUI.
    """
    global dark_mode
    dark_mode = dark_mode_var.get()
    if dark_mode:
        bg_color = "black"
        fg_color = "white"
        insertbackground_color = "cyan"
    else:
        bg_color = "white"
        fg_color = "black"
        insertbackground_color = "black"

    root_list = [root, sci_tab, hk_tab]
    sci_tab.configure(bg=bg_color)
    hk_tab.configure(bg=bg_color)
    root.configure(bg=bg_color)

    for root_item in root_list:
        # try to change the background and foreground colors of the root item
        try:
            root_item.configure(bg=bg_color)
        except Exception:
            pass
        try:
            root_item.configure(fg=fg_color)
        except Exception:
            pass

        # try to chhange the color of the buttons
        for button in root_item.winfo_children():
            try:
                button.configure(bg=bg_color, fg=fg_color)
            except Exception:
                pass

        # try to change the color of the text boxes
        for textbox in root_item.winfo_children():
            try:
                textbox.configure(bg=bg_color, fg=fg_color, insertbackground=insertbackground_color)
            except Exception:
                pass

        # Change the color of the labels
        for label in root_item.winfo_children():
            try:
                label.configure(bg=bg_color, fg=fg_color)
            except Exception:
                pass
        # Select the dropdown boxes and change the color
        for dropdown in root_item.winfo_children():
            try:
                dropdown.configure(bg=bg_color, fg=fg_color)
            except Exception:
                pass
        # Skip the Quit button
        for button in root_item.winfo_children():
            try:
                if button["text"] == "Quit":
                    button.configure(fg="red")
            except Exception:
                pass
        for button in root_item.winfo_children():
            try:
                if button["text"] in ["Dark Mode", "Default Options", "Load Files", "Refresh",
                                      "Save CDF", "Save CSV"]:
                    button.configure(fg="green")
            except Exception:
                pass

    # Check if global_variables.all_file_details is empty. If not, then refresh the plots
    if global_variables.all_file_details:
        try:
            refresh_ts_plot()
        except Exception:
            pass

        try:
            hist_plot_inputs()
        except Exception:
            pass

# Create the main window.
root = tk.Tk()

# Get the DPI of the screen. This is used to scale the figure size.
dpi = root.winfo_fpixels("1i")

# NOTE: This hack is necessary since I am using multiple monitors. This can be edited as we work on
# a different machine.
# Check whether the operating system is windows or linux, and assign the correct screen width and
# height.
if platform.system() == "Windows":
    screen_width, screen_height = (
        0.9 * root.winfo_screenwidth(),
        0.9 * root.winfo_screenheight(),
    )
if platform.system() == "Linux":
    screen_width, screen_height = (
        0.45 * root.winfo_screenwidth(),
        0.8 * root.winfo_screenheight(),
    )
else:
    screen_width, screen_height = (
        0.9 * root.winfo_screenwidth(),
        0.9 * root.winfo_screenheight(),
    )

screen_width = 1200
screen_height = 800
# print(
#     "If the GUI size is messed up, check comment on line #215 of the code 'lxi_gui.py'."
# )

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

tabControl.add(sci_tab, text="Science Tab")
tabControl.add(hk_tab, text="Housekeeping Tab")

# Configure the science tab rows and columns.
sci_tab.columnconfigure(0, {"minsize": 1}, weight=1)
sci_tab.columnconfigure(1, {"minsize": 1}, weight=2)
sci_tab.columnconfigure(2, {"minsize": 1}, weight=5)
sci_tab.columnconfigure(3, {"minsize": 1}, weight=5)
sci_tab.columnconfigure(4, {"minsize": 1}, weight=1)
sci_tab.columnconfigure(5, {"minsize": 1}, weight=1)
sci_tab.columnconfigure(6, {"minsize": 1}, weight=1)
sci_tab.columnconfigure(7, {"minsize": 1}, weight=1)
sci_tab.columnconfigure(8, {"minsize": 1}, weight=1)
sci_tab.columnconfigure(9, {"minsize": 1}, weight=1)

# Configure the sci_tab rows
# for i in range(0, 20):
#     sci_tab.rowconfigure(i, {'minsize': 0}, weight=0)
# Choose a font style for GUI
font_style = font.Font(family="serif", size=12)
font_style_box = font.Font(family="serif", size=12, weight="bold")
font_style_big = font.Font(family="serif", size=25)

dark_mode = True
if dark_mode:
    bg_color = "black"
    fg_color = "white"
    insertbackground_color = "cyan"
else:
    bg_color = "white"
    fg_color = "black"
    insertbackground_color = "black"

# Add a checkbutton to enable/disable dark mode
dark_mode_var = tk.BooleanVar()
dark_mode_var.set(dark_mode)
dark_mode_button = tk.Checkbutton(
    sci_tab,
    text="Dark Mode",
    variable=dark_mode_var,
    command=dark_mode_change,
    bg=bg_color,
    fg=fg_color,
    font=font_style,
    relief="raised",
    highlightthickness=5,
    highlightcolor=bg_color,
    selectcolor="#808080",
    cursor="hand2",
)
dark_mode_button.grid(row=19, column=0, sticky="nsew", padx=5, pady=5)
# add this button on the housekeeping tab as well
dark_mode_button = tk.Checkbutton(
    hk_tab,
    text="Dark Mode",
    variable=dark_mode_var,
    command=dark_mode_change,
    bg=bg_color,
    fg=fg_color,
    font=font_style,
    relief="raised",
    highlightthickness=5,
    highlightcolor=bg_color,
    selectcolor="#808080",
    cursor="hand2",
)
dark_mode_button.grid(row=12, column=6, columnspan=1, sticky="nsew", padx=5, pady=5)


sci_tab.configure(
    bg=bg_color, padx=5, pady=5, relief="raised", borderwidth=5, highlightthickness=5
)
hk_tab.configure(bg=bg_color, padx=5, pady=5, relief="raised", borderwidth=5, highlightthickness=5)

# Configure the housekeeping tab rows and columns.
for i in range(0, 10):
    hk_tab.rowconfigure(i, {"minsize": 1}, weight=1)
    hk_tab.columnconfigure(i, {"minsize": 1}, weight=1)

# Insert a file load button
# For science file
sci_file_load_button = tk.Button(
    sci_tab, text="Load Science File", command=lxrf.open_file_sci, font=font_style, bg=bg_color,
    fg=fg_color
)
sci_file_load_button.grid(row=0, column=0, columnspan=1, pady=0, sticky="ew")

sci_file_name = tk.StringVar()
sci_file_name.set("No file loaded")
sci_file_load_entry = tk.Entry(
    sci_tab,
    textvariable=sci_file_name,
    font=font_style,
    justify="left",
    bg=bg_color,
    fg="red",
    relief="sunken",
    borderwidth=2,
)
sci_file_load_entry.grid(row=1, column=0, columnspan=2, pady=0, sticky="ew")
sci_file_load_entry.config(state="disabled", disabledbackground="black", disabledforeground="gray")

# insert the file_load_entry value into the entry box only if the sci_file_load_button is clicked
sci_file_load_button.config(
    command=lambda: sci_file_load_entry.insert(0, lxrf.open_file_sci())
)
sci_file_load_button.config(state="disabled")

# For housekeeping file
hk_file_load_button = tk.Button(
    sci_tab, text="Load HK File", command=lxrf.open_file_hk, font=font_style, bg=bg_color, fg=fg_color
)
hk_file_load_button.grid(row=2, column=0, columnspan=1, pady=0, sticky="ew")

hk_file_name = tk.StringVar()
hk_file_name.set("No file loaded")
hk_file_load_entry = tk.Entry(
    sci_tab,
    textvariable=hk_file_name,
    font=font_style,
    justify="left",
    bg=bg_color,
    fg="red",
    relief="sunken",
    borderwidth=2,
)
hk_file_load_entry.grid(row=3, column=0, columnspan=2, pady=0, sticky="ew")
hk_file_load_entry.config(state="disabled", disabledbackground="black", disabledforeground="gray")

# insert the file_load_entry value into the entry box only if the hk_file_load_button is clicked
hk_file_load_button.config(
    command=lambda: hk_file_load_entry.insert(0, lxrf.open_file_hk())
)
hk_file_load_button.config(state="disabled")

# For binary file
b_file_load_button = tk.Button(
    sci_tab, text="Load binary File", command=lxrf.open_file_b, font=font_style, bg=bg_color, fg=fg_color
)
b_file_load_button.grid(row=4, column=0, columnspan=1, pady=0, sticky="ew")

b_file_name = tk.StringVar()
b_file_name.set("No file loaded")
b_file_load_entry = tk.Entry(
    sci_tab,
    textvariable=b_file_name,
    font=font_style,
    justify="left",
    bg=bg_color,
    fg="red",
    relief="sunken",
    borderwidth=2,
)
b_file_load_entry.grid(row=5, column=0, columnspan=2, pady=0, sticky="ew")
b_file_load_entry.config(state="disabled", disabledbackground="black", disabledforeground="gray")

# insert the file_load_entry value into the entry box only if the b_file_load_button is clicked
b_file_load_button.config(
    command=lambda: b_file_load_entry.insert(
        0, lxrf.open_file_b(t_start=start_time.get(), t_end=end_time.get())
    )
)
b_file_load_button.config(state="disabled")

# If a new file is loaded, then print its name in the entry box and update the file_name variable.
sci_file_name.trace(
    "w", lambda *_: sci_file_name.set(lmsc.file_name_update(file_type="sci"))
)
hk_file_name.trace(
    "w", lambda *_: hk_file_name.set(lmsc.file_name_update(file_type="hk"))
)

# If a new binary file is loaded, then update the name of all three files.
b_file_name.trace("w", lambda *_: b_file_name.set(lmsc.file_name_update(file_type="b")))
b_file_name.trace(
    "w", lambda *_: sci_file_name.set(lmsc.file_name_update(file_type="sci"))
)
b_file_name.trace(
    "w", lambda *_: hk_file_name.set(lmsc.file_name_update(file_type="hk"))
)

# If the global_variables.all_file_details["df_slice_hk"] is not empty, then set the comlumn names
# to the columns in the dataframe
if bool("df_slice_hk" in global_variables.all_file_details.keys()):
    ts_options = global_variables.all_file_details["df_slice_hk"].columns.tolist()
else:
    ts_options = [
        "HK_id",
        "PinPullerTemp",
        "OpticsTemp",
        "LEXIbaseTemp",
        "HVsupplyTemp",
        "+5.2V_Imon",
        "+10V_Imon",
        "+3.3V_Imon",
        "AnodeVoltMon",
        "+28V_Imon",
        "ADC_Ground",
        "Cmd_count",
        "Pinpuller_Armed",
        "HVmcpAuto",
        "HVmcpMan",
        "DeltaEvntCount",
        "DeltaDroppedCount",
        "DeltaLostEvntCount",
    ]

# Plot options for the first plot
plot_opt_label_1 = tk.Label(hk_tab, text="Plot options:", font=font_style_box)
plot_opt_label_1.grid(row=0, column=0, columnspan=1, pady=0, sticky="w")
plot_opt_label_1.config(fg=fg_color, bg=bg_color)

plot_opt_entry_1 = tk.StringVar(hk_tab)
plot_opt_entry_1.set("Select a column")
ts_menu_1 = tk.OptionMenu(hk_tab, plot_opt_entry_1, *ts_options)
ts_menu_1.config(fg=fg_color, bg=bg_color)
ts_menu_1.grid(row=0, column=2, columnspan=1, sticky="w")

# Plot options for the second plot
plot_opt_entry_2 = tk.StringVar(hk_tab)
plot_opt_entry_2.set("Select a column")
ts_menu_2 = tk.OptionMenu(hk_tab, plot_opt_entry_2, *ts_options)
ts_menu_2.config(fg=fg_color, bg=bg_color)
ts_menu_2.grid(row=0, column=5, columnspan=1, sticky="w")

# Plot optiosn for third plot
plot_opt_entry_3 = tk.StringVar(hk_tab)
plot_opt_entry_3.set("Select a column")
ts_menu_3 = tk.OptionMenu(hk_tab, plot_opt_entry_3, *ts_options)
ts_menu_3.config(fg=fg_color, bg=bg_color)
ts_menu_3.grid(row=0, column=8, columnspan=1, sticky="w")

# Plot options for fourth plot (in the second row)
plot_opt_entry_4 = tk.StringVar(hk_tab)
plot_opt_entry_4.set("Select a column")
ts_menu_4 = tk.OptionMenu(hk_tab, plot_opt_entry_4, *ts_options)
ts_menu_4.config(fg=fg_color, bg=bg_color)
ts_menu_4.grid(row=2, column=2, columnspan=1, sticky="w")

# Plot options for fifth plot (in the second row)
plot_opt_entry_5 = tk.StringVar(hk_tab)
plot_opt_entry_5.set("Select a column")
ts_menu_5 = tk.OptionMenu(hk_tab, plot_opt_entry_5, *ts_options)
ts_menu_5.config(fg=fg_color, bg=bg_color)
ts_menu_5.grid(row=2, column=5, columnspan=1, sticky="w")

# Plot options for sixth plot (in the second row)
plot_opt_entry_6 = tk.StringVar(hk_tab)
plot_opt_entry_6.set("Select a column")
ts_menu_6 = tk.OptionMenu(hk_tab, plot_opt_entry_6, *ts_options)
ts_menu_6.config(fg=fg_color, bg=bg_color)
ts_menu_6.grid(row=2, column=8, columnspan=1, sticky="w")

# Plot options for seventh plot (in the third row)
plot_opt_entry_7 = tk.StringVar(hk_tab)
plot_opt_entry_7.set("Select a column")
ts_menu_7 = tk.OptionMenu(hk_tab, plot_opt_entry_7, *ts_options)
ts_menu_7.config(fg=fg_color, bg=bg_color)
ts_menu_7.grid(row=4, column=2, columnspan=1, sticky="w")

# Plot options for eighth plot (in the third row)
plot_opt_entry_8 = tk.StringVar(hk_tab)
plot_opt_entry_8.set("Select a column")
ts_menu_8 = tk.OptionMenu(hk_tab, plot_opt_entry_8, *ts_options)
ts_menu_8.config(fg=fg_color, bg=bg_color)
ts_menu_8.grid(row=4, column=5, columnspan=1, sticky="w")

# Plot options for ninth plot (in the third row)
plot_opt_entry_9 = tk.StringVar(hk_tab)
plot_opt_entry_9.set("Select a column")
ts_menu_9 = tk.OptionMenu(hk_tab, plot_opt_entry_9, *ts_options)
ts_menu_9.config(fg=fg_color, bg=bg_color)
ts_menu_9.grid(row=4, column=8, columnspan=1, sticky="w")

(
    x_min_entry,
    x_max_entry,
    y_min_entry,
    y_max_entry,
    hist_bins_entry,
    c_min_entry,
    c_max_entry,
    density_status_var,
    norm_type_var,
    unit_type_var,
    v_min_thresh_entry,
    v_max_thresh_entry,
    v_sum_min_thresh_entry,
    v_sum_max_thresh_entry,
    cut_status_var,
    curve_fit_status_var,
    lin_corr_status_var,
    cmap_option,
) = lgeb.populate_entries(root=sci_tab, dark_mode=dark_mode)

# Redo the histogram plot if the status of the checkbox is changed
density_status_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

# Redo the histogram plot when the norm type is changed
norm_type_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

# Redo the histogram plot when the unit type is changed
unit_type_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

cut_status_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

curve_fit_status_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

lin_corr_status_var.trace("w", lambda *_: hist_plot_inputs(dpi=dpi))

# Add a button to save the data to a cdf file
cdf_save_button = tk.Button(
    sci_tab,
    text="Save CDF",
    command=lambda: lmsc.save_cdf(),
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg="green",
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="green",
    highlightcolor="green",
)
cdf_save_button.grid(row=18, column=11, columnspan=1, sticky="nw")
# Disable the button until the data is loaded
cdf_save_button.config(state="normal")

# Add a button to save the data to a csv file
csv_save_button = tk.Button(
    sci_tab,
    text="Save CSV",
    command=lambda: lmsc.save_csv(),
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg="green",
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="green",
    highlightcolor="green",
)
csv_save_button.grid(row=18, column=12, columnspan=1, sticky="nw")

# Add a checkox to enable/disable the multiple file selection option
multi_file_status_var = tk.IntVar()
multi_file_status_var.set(1)
multi_file_status = tk.Checkbutton(
    sci_tab,
    text="Multiple Files",
    variable=multi_file_status_var,
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg=fg_color,
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="black",
    highlightcolor=bg_color,
    selectcolor="#808080",
    cursor="hand2",
)
multi_file_status.grid(row=6, column=0, columnspan=1, sticky="nw")

# If the multiple file status is selected, then disable the "sci_file_load_button",
# "hk_file_load_button", and "b_file_load_button" buttons
multi_file_status_var.trace(
    "w", lambda *_: lmsc.change_state(button=sci_file_load_button)
)
multi_file_status_var.trace(
    "w", lambda *_: lmsc.change_state(button=sci_file_load_entry)
)
multi_file_status_var.trace(
    "w", lambda *_: lmsc.change_state(button=hk_file_load_button)
)
multi_file_status_var.trace(
    "w", lambda *_: lmsc.change_state(button=hk_file_load_entry)
)
multi_file_status_var.trace(
    "w", lambda *_: lmsc.change_state(button=b_file_load_button)
)
multi_file_status_var.trace("w", lambda *_: lmsc.change_state(button=b_file_load_entry))
multi_file_status_var.trace(
    "w", lambda *_: lmsc.change_state(button=folder_load_button)
)
multi_file_status_var.trace("w", lambda *_: lmsc.change_state(button=folder_path))

# Add a text box to enter the folder path
folder_path = tk.Entry(sci_tab, justify="center", bg=bg_color, fg="green", borderwidth=2)
folder_path.grid(row=7, column=0, columnspan=2, sticky="nsew")

# Set the default folder name in the text box
# folder_path.insert(1, "For multiple files, enter the folder path here")
# Insert the default folder path in the text box based on the operating system
if os.name == "nt":
    folder_path.insert(
        1, "C:\\Users\\Lexi-User\\Desktop\\PIT_softwares\\PIT_23_05_05\\Target\\rec_tlm\\not_sent\\"
    )
elif os.name == "posix":
    folder_path.insert(1, "/home/vetinari/Desktop/git/Lexi-Bu/lxi_gui/data/PIT/20230608_not_sent/")
elif os.name == "darwin":
    folder_path.insert(1, "/Users/lexi_user/Desktop/PIT_softwares/PIT_23_05_05/Target/rec_tlm/not_sent/")
else:
    raise OSError("Operating system not supported")

folder_path.config(insertbackground=insertbackground_color)
folder_path.config(state="normal", disabledbackground="black", disabledforeground="gray")

# Add a button to load all the files in the folder_path
folder_load_button = tk.Button(
    sci_tab,
    text="Load Files",
    command=lambda: load_and_copy_files(),
    # command=lambda: lmsc.load_folder(
    #     file_val=folder_path.get(),
    #     t_start=start_time.get(),
    #     t_end=end_time.get(),
    #     multiple_files=multi_file_status_var.get(),
    # ),
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg="green",
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="green",
    highlightcolor="green",
)
folder_load_button.grid(row=6, column=1, columnspan=1, sticky="nw")
folder_load_button.config(state="normal")

# Add the load files button to hk tab as well
folder_load_button_hk = tk.Button(
    hk_tab,
    text="Load Files",
    command=lambda: load_and_copy_files(),
    # command=lambda: lmsc.load_folder(
    #     file_val=folder_path.get(),
    #     t_start=start_time.get(),
    #     t_end=end_time.get(),
    #     multiple_files=multi_file_status_var.get(),
    # ),
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg="green",
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="green",
    highlightcolor="green",
)
folder_load_button_hk.grid(row=12, column=7, columnspan=1, sticky="nsew")
folder_load_button_hk.config(state="normal")

# Label for plot times
start_time_label = tk.Label(
    sci_tab, text="Plot Times", font=font_style, bg=bg_color, fg=fg_color
)
start_time_label.grid(row=8, column=0, columnspan=2, sticky="nsew")

# Add an input box with a label for start time
default_time_dict = lgcf.get_config_time()
start_time = tk.Entry(sci_tab, justify="center", bg=bg_color, fg="green", borderwidth=2)
start_time.insert(0, default_time_dict["start_time"])
start_time.config(insertbackground=insertbackground_color)
start_time.grid(row=9, column=0, columnspan=2, sticky="nsew")
start_time_label = tk.Label(
    sci_tab, text="Start Time", font=font_style, bg=bg_color, fg=fg_color
)
start_time_label.grid(row=10, column=0, columnspan=2, sticky="nsew")

end_time = tk.Entry(sci_tab, justify="center", bg=bg_color, fg="green", borderwidth=2)
end_time.insert(0, default_time_dict["end_time"])
end_time.config(insertbackground=insertbackground_color)
end_time.grid(row=11, column=0, columnspan=2, sticky="nsew")
end_time_label = tk.Label(
    sci_tab, text="End Time", font=font_style, bg=bg_color, fg=fg_color
)
end_time_label.grid(row=12, column=0, columnspan=2)

# Add the start and end time to hk tab as well
start_time_hk = tk.Entry(hk_tab, justify="center", bg=bg_color, fg="green", borderwidth=2)
start_time_hk.insert(0, default_time_dict["start_time"])
start_time_hk.config(insertbackground=insertbackground_color)
start_time_hk.grid(row=12, column=1, columnspan=1, sticky="nsew")
start_time_label_hk = tk.Label(
    hk_tab, text="Start Time", font=font_style, bg=bg_color, fg=fg_color
)
start_time_label_hk.grid(row=13, column=1, columnspan=1, sticky="nsew")

end_time_hk = tk.Entry(hk_tab, justify="center", bg=bg_color, fg="green", borderwidth=2)
end_time_hk.insert(0, default_time_dict["end_time"])
end_time_hk.config(insertbackground=insertbackground_color)
end_time_hk.grid(row=12, column=4, columnspan=1, sticky="nsew")
end_time_label_hk = tk.Label(
    hk_tab, text="End Time", font=font_style, bg=bg_color, fg=fg_color
)
end_time_label_hk.grid(row=13, column=4, columnspan=1, sticky="nsew")

# If the start time in one tab is changed, update the other tab
start_time.bind("<KeyRelease>", lambda *_: update_time_entry(start_time, start_time_hk))
start_time_hk.bind("<KeyRelease>", lambda *_: update_time_entry(start_time_hk, start_time))

# If the end time in one tab is changed, update the other tab
end_time.bind("<KeyRelease>", lambda *_: update_time_entry(end_time, end_time_hk))
end_time_hk.bind("<KeyRelease>", lambda *_: update_time_entry(end_time_hk, end_time))

# Add a dropdown menu for the kind of time to be used, options are "Lexi Time" and "UTC Time" and
# "Local Time"
# Default is "Lexi Time". If the time kind is changed, then run the "refresh_ts_plot" function
# to update the plot.
time_type = tk.StringVar()
time_type.set("LEXI")
time_type_menu = tk.OptionMenu(
    hk_tab,
    time_type,
    "LEXI",
    "UTC",
    "Local",
    command=lambda *_: refresh_ts_plot(),
)
# Deactivate the UTC and Local options for now
time_type_menu["menu"].entryconfig(1, state="disabled")
time_type_menu["menu"].entryconfig(2, state="disabled")

time_type_menu.config(bg=bg_color, fg=fg_color, borderwidth=2)
time_type_menu.grid(row=12, column=2, columnspan=1, sticky="nsew")
time_type_label = tk.Label(
    hk_tab, text="Time Type", font=font_style, bg=bg_color, fg=fg_color
)
time_type_label.grid(row=13, column=2, columnspan=1, sticky="nsew")

# if any of the ts_options are changed, update the plot
plot_opt_entry_1.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_1, row=1, column=0, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_2.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_2, row=1, column=3, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_3.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_3, row=1, column=6, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_4.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_4, row=3, column=0, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_5.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_5, row=3, column=3, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_6.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_6, row=3, column=6, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_7.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_7, row=5, column=0, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_8.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_8, row=5, column=3, rowspan=1, columnspan=3
    ),
)

plot_opt_entry_9.trace(
    "w",
    lambda *_: ts_plot_inputs(
        plot_opt_entry=plot_opt_entry_9, row=5, column=6, rowspan=1, columnspan=3
    ),
)

# If the plot button is pressed then all the histogram plots are redrawn
plot_button = tk.Button(
    sci_tab,
    bg=bg_color,
    fg=fg_color,
    text="Plot Histogram",
    font=font_style_box,
    justify="center",
    command=lambda: hist_plot_inputs(dpi=dpi),
)

plot_button.grid(
    row=13, column=0, columnspan=1, rowspan=1, sticky="nsew", pady=5, padx=5
)

# If the plot button is pressed, then print the current time
plot_button.bind(
    "<Button-1>",
    lambda event: lmsc.print_time_details(
        start_time=start_time.get(), end_time=end_time.get()
    ),
)

# If the plot button is pressed then all the histogram plots are redrawn
refresh_button = tk.Button(
    sci_tab,
    bg=bg_color,
    fg=fg_color,
    text="Refresh Histogram",
    font=font_style_box,
    justify="center",
    command=lambda: hist_plot_inputs(dpi=dpi),
)

refresh_button.grid(
    row=14, column=0, columnspan=1, rowspan=1, sticky="nsew", pady=5, padx=5
)

# If the plot button is pressed, then print the current time
refresh_button.bind(
    "<Button-1>",
    lambda event: lmsc.print_time_details(
        start_time=start_time.get(), end_time=end_time.get()
    ),
)

# Add a button to save the configuration file
entry_list = [
    x_min_entry,
    x_max_entry,
    y_min_entry,
    y_max_entry,
    hist_bins_entry,
    c_min_entry,
    c_max_entry,
    density_status_var,
    norm_type_var,
    unit_type_var,
    v_min_thresh_entry,
    v_max_thresh_entry,
    v_sum_min_thresh_entry,
    v_sum_max_thresh_entry,
    cut_status_var,
    curve_fit_status_var,
    lin_corr_status_var,
    cmap_option,
    start_time,
    end_time,
]

save_config_button = tk.Button(
    sci_tab,
    bg=bg_color,
    fg=fg_color,
    text="Save Config",
    font=font_style_box,
    justify="center",
    command=lambda: lgcf.save_config(
        entry_list=entry_list, entry_sec=["sci_plot_options", "time_options"]
    ),
)
save_config_button.grid(
    row=15, column=0, columnspan=1, rowspan=1, sticky="nsew", pady=5, padx=5
)

# FIXME: Default config button doesn't work
# Add a default button to reset the configuration file
default_config_button = tk.Button(
    sci_tab,
    bg=bg_color,
    fg=fg_color,
    text="Default Config",
    font=font_style_box,
    justify="center",
    command=lambda: lgcf.create_config_file(default_vals=True),
)
default_config_button.grid(
    row=16, column=0, columnspan=1, rowspan=1, sticky="nsew", pady=5, padx=5
)


# Dsiable the default config button
default_config_button.config(state="disabled", disabledforeground="grey", relief="sunken")

# Add a quit button
quit_button_sci = tk.Button(
    sci_tab,
    text="Quit",
    command=root.destroy,
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg="red",
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="red",
    highlightcolor="red",
)
quit_button_sci.grid(row=18, column=13, columnspan=1, rowspan=1, sticky="ne")

# Add a default option check box
default_opt_var = tk.BooleanVar()
default_opt_var.set(True)
default_opt_checkbox = tk.Checkbutton(
    hk_tab,
    text="Default Options",
    font=font_style_box,
    variable=default_opt_var,
    bg=bg_color,
    fg=fg_color,
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=5,
    highlightcolor=bg_color,
    selectcolor="#808080",
    cursor="hand2",
)
default_opt_checkbox.grid(row=12, column=0, columnspan=1, sticky="nw")

default_opt_var.trace("w", lambda *_: ts_button_val_change(default_opt_var))

# Add a clickable button to display the time values in the terminal
time_button = tk.Button(
    sci_tab,
    bg=bg_color,
    fg=fg_color,
    text="Print Time",
    font=font_style_box,
    justify="center",
    command=lambda: lmsc.print_time_details(
        start_time=start_time.get(), end_time=end_time.get()
    ),
)
time_button.grid(
    row=17, column=0, columnspan=1, rowspan=1, sticky="nsew", pady=5, padx=5
)

copy_button = tk.Button(
    sci_tab,
    bg=bg_color,
    fg=fg_color,
    text="Copy PIT Files",
    font=font_style_box,
    justify="center",
    command=lambda: lmsc.copy_pit_files(),
)
copy_button.grid(
    row=18, column=0, columnspan=1, rowspan=1, sticky="nsew", pady=5, padx=5
)

# Add a refresh button to reload all the time series plots
refresh_ts_hk_button = tk.Button(
    hk_tab,
    text="Refresh",
    command=lambda: refresh_ts_plot(),
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg="green",
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="green",
    highlightcolor="green",
)
refresh_ts_hk_button.grid(row=12, column=8, columnspan=1, rowspan=1, sticky="new")

quit_button_hk = tk.Button(
    hk_tab,
    text="Quit",
    command=root.destroy,
    font=font_style_box,
    justify="center",
    bg=bg_color,
    fg="red",
    pady=5,
    padx=5,
    borderwidth=2,
    relief="raised",
    highlightthickness=2,
    highlightbackground="red",
    highlightcolor="red",
)
quit_button_hk.grid(row=12, column=9, columnspan=1, rowspan=1, sticky="new")

root.mainloop()
