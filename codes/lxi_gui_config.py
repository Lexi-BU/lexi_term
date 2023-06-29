import os
from configparser import ConfigParser


def create_config_file(default_vals=False):
    """Create a configuration file if it does not exist."""
    # If the configuration file doesn't exist, then create a new one
    # with default options
    if not default_vals:
        if not os.path.isfile("luigi.cfg"):
            gui_config = ConfigParser()
            gui_config.add_section("sci_plot_options")
            gui_config.set("sci_plot_options", "x_min_entry", "0")
            gui_config.set("sci_plot_options", "x_max_entry", "0")
            gui_config.set("sci_plot_options", "y_min_entry", "0")
            gui_config.set("sci_plot_options", "y_max_entry", "0")
            gui_config.set("sci_plot_options", "hist_bin_entry", "0")
            gui_config.set("sci_plot_options", "c_min_entry", "0")
            gui_config.set("sci_plot_options", "c_max_entry", "0")
            gui_config.set("sci_plot_options", "density_status", "0")
            gui_config.set("sci_plot_options", "norm_type", "0")
            gui_config.set("sci_plot_options", "unit_type", "0")
            gui_config.set("sci_plot_options", "v_min_thresh_entry", "0")
            gui_config.set("sci_plot_options", "v_max_thresh_entry", "0")
            gui_config.set("sci_plot_options", "v_sum_min_thresh_entry", "0")
            gui_config.set("sci_plot_options", "v_sum_max_thresh_entry", "0")
            gui_config.set("sci_plot_options", "cut_status", "0")
            gui_config.set("sci_plot_options", "curve_fit_status", "0")
            gui_config.set("sci_plot_options", "lin_corr_status", "0")
            gui_config.set("sci_plot_options", "cmap", "0")

            gui_config.add_section("time_options")
            gui_config.set("time_options", "start_time", "YYYY-MM-DD HH:MM:SS")
            gui_config.set("time_options", "end_time", "YYYY-MM-DD HH:MM:SS")
            with open("luigi.cfg", "w") as config_file:
                gui_config.write(config_file)
            print("\033[1;32mConfiguration file 'luigi.cfg' created.\033[0m")
    elif default_vals:
        gui_config = ConfigParser()
        gui_config.add_section("sci_plot_options")
        gui_config.set("sci_plot_options", "x_min_entry", "-5")
        gui_config.set("sci_plot_options", "x_max_entry", "5")
        gui_config.set("sci_plot_options", "y_min_entry", "-5")
        gui_config.set("sci_plot_options", "y_max_entry", "5")
        gui_config.set("sci_plot_options", "hist_bin_entry", "200")
        gui_config.set("sci_plot_options", "c_min_entry", "1")
        gui_config.set("sci_plot_options", "c_max_entry", "None")
        gui_config.set("sci_plot_options", "density_status", "0")
        gui_config.set("sci_plot_options", "norm_type", "log")
        gui_config.set("sci_plot_options", "unit_type", "mcp")
        gui_config.set("sci_plot_options", "v_min_thresh_entry", "1.2")
        gui_config.set("sci_plot_options", "v_max_thresh_entry", "3.4")
        gui_config.set("sci_plot_options", "v_sum_min_thresh_entry", "5")
        gui_config.set("sci_plot_options", "v_sum_max_thresh_entry", "7")
        gui_config.set("sci_plot_options", "cut_status", "0")
        gui_config.set("sci_plot_options", "curve_fit_status", "0")
        gui_config.set("sci_plot_options", "lin_corr_status", "1")
        gui_config.set("sci_plot_options", "cmap", "viridis")

        gui_config.add_section("time_options")
        gui_config.set("time_options", "start_time", "YYYY-MM-DD HH:MM:SS")
        gui_config.set("time_options", "end_time", "YYYY-MM-DD HH:MM:SS")
        with open("luigi.cfg", "w") as config_file:
            gui_config.write(config_file)
    return None


def get_config_entry(default_vals=False):
    """Get the configuration file entries"""
    # Default plot options
    # Check if a configuration file exists with extension .cfg
    # If it does, then load the configuration file
    # If it does not, then create a new configuration file
    create_config_file(default_vals=default_vals)
    entry_sec = "sci_plot_options"
    # if os.path.isfile("luigi.cfg"):
    gui_config = ConfigParser()
    gui_config.read("luigi.cfg")
    # Check if the section exists
    if gui_config.has_section(entry_sec):
        default_opt_dict = {
            "x_min_entry": gui_config.get(entry_sec, "x_min_entry"),
            "x_max_entry": gui_config.get(entry_sec, "x_max_entry"),
            "y_min_entry": gui_config.get(entry_sec, "y_min_entry"),
            "y_max_entry": gui_config.get(entry_sec, "y_max_entry"),
            "hist_bin_entry": gui_config.get(entry_sec, "hist_bin_entry"),
            "c_min_entry": gui_config.get(entry_sec, "c_min_entry"),
            "c_max_entry": gui_config.get(entry_sec, "c_max_entry"),
            "density_status": gui_config.get(entry_sec, "density_status"),
            "norm_type": gui_config.get(entry_sec, "norm_type"),
            "unit_type": gui_config.get(entry_sec, "unit_type"),
            "v_min_thresh_entry": gui_config.get(entry_sec, "v_min_thresh_entry"),
            "v_max_thresh_entry": gui_config.get(entry_sec, "v_max_thresh_entry"),
            "v_sum_min_thresh_entry": gui_config.get(
                entry_sec, "v_sum_min_thresh_entry"
            ),
            "v_sum_max_thresh_entry": gui_config.get(
                entry_sec, "v_sum_max_thresh_entry"
            ),
            "cut_status": gui_config.get(entry_sec, "cut_status"),
            "curve_fit_status": gui_config.get(entry_sec, "curve_fit_status"),
            "lin_corr_status": gui_config.get(entry_sec, "lin_corr_status"),
            "cmap": gui_config.get(entry_sec, "cmap"),
        }
    else:
        default_opt_dict = {
            "x_min_entry": "0.41",
            "x_max_entry": "0.65",
            "y_min_entry": "0.45",
            "y_max_entry": "0.65",
            "hist_bin_entry": "200",
            "c_min_entry": "1",
            "c_max_entry": "5",
            "density_status": "False",
            "norm_type": "linear",
            "unit_type": "volt",
            "v_min_thresh_entry": "1.2",
            "v_max_thresh_entry": "3.4",
            "v_sum_min_thresh_entry": "5",
            "v_sum_max_thresh_entry": "6",
            "cut_status": "False",
            "curve_fit_status": "False",
            "lin_corr_status": "False",
            "cmap": "viridis",
        }
        # Save the default options to a configuration file
        gui_config = ConfigParser()
        gui_config.add_section(entry_sec)
        gui_config.set(entry_sec, "x_min_entry", default_opt_dict["x_min_entry"])
        gui_config.set(entry_sec, "x_max_entry", default_opt_dict["x_max_entry"])
        gui_config.set(entry_sec, "y_min_entry", default_opt_dict["y_min_entry"])
        gui_config.set(entry_sec, "y_max_entry", default_opt_dict["y_max_entry"])
        gui_config.set(entry_sec, "hist_bin_entry", default_opt_dict["hist_bin_entry"])
        gui_config.set(entry_sec, "c_min_entry", default_opt_dict["c_min_entry"])
        gui_config.set(entry_sec, "c_max_entry", default_opt_dict["c_max_entry"])
        gui_config.set(entry_sec, "density_status", default_opt_dict["density_status"])
        gui_config.set(entry_sec, "norm_type", default_opt_dict["norm_type"])
        gui_config.set(entry_sec, "unit_type", default_opt_dict["unit_type"])
        gui_config.set(
            entry_sec, "v_min_thresh_entry", default_opt_dict["v_min_thresh_entry"]
        )
        gui_config.set(
            entry_sec, "v_max_thresh_entry", default_opt_dict["v_max_thresh_entry"]
        )
        gui_config.set(
            entry_sec,
            "v_sum_min_thresh_entry",
            default_opt_dict["v_sum_min_thresh_entry"],
        )
        gui_config.set(
            entry_sec,
            "v_sum_max_thresh_entry",
            default_opt_dict["v_sum_max_thresh_entry"],
        )
        gui_config.set(entry_sec, "cut_status", default_opt_dict["cut_status"])
        gui_config.set(
            entry_sec, "curve_fit_status", default_opt_dict["curve_fit_status"]
        )
        gui_config.set(
            entry_sec, "lin_corr_status", default_opt_dict["lin_corr_status"]
        )
        gui_config.set(entry_sec, "cmap", default_opt_dict["cmap"])
        with open("luigi.cfg", "w") as config_file:
            gui_config.write(config_file)
        print("\033[1;32mConfiguration file 'luigi.cfg' created.\033[0m")

    return default_opt_dict


def get_config_time():
    """Get the time of the last configuration file modification."""
    entry_sec = "time_options"
    create_config_file()
    gui_config = ConfigParser()
    gui_config.read("luigi.cfg")
    # Check if the section exists
    if gui_config.has_section(entry_sec):
        default_opt_dict = {
            "start_time": gui_config.get(entry_sec, "start_time"),
            "end_time": gui_config.get(entry_sec, "end_time"),
        }
    else:
        default_opt_dict = {
            "start_time": "YYYY-MM-DD HH:MM:SS",
            "end_time": "YYYY-MM-DD HH:MM:SS",
        }
        # Save the default options to a configuration file
        gui_config.add_section(entry_sec)
        gui_config.set(entry_sec, "start_time", default_opt_dict["start_time"])
        gui_config.set(entry_sec, "end_time", default_opt_dict["end_time"])
        with open("luigi.cfg", "w") as config_file:
            gui_config.write(config_file)
        print("\033[1;32mConfiguration file 'luigi.cfg' created.\033[0m")
    return default_opt_dict


def save_config(entry_list=None, entry_sec=["sci_plot_options"]):
    """Save the current options to a configuration file."""

    config_vals = [str(entry.get()) for entry in entry_list]
    gui_config = ConfigParser()
    gui_config.add_section(entry_sec[0])
    gui_config.set(entry_sec[0], "x_min_entry", config_vals[0])
    gui_config.set(entry_sec[0], "x_max_entry", config_vals[1])
    gui_config.set(entry_sec[0], "y_min_entry", config_vals[2])
    gui_config.set(entry_sec[0], "y_max_entry", config_vals[3])
    gui_config.set(entry_sec[0], "hist_bin_entry", config_vals[4])
    gui_config.set(entry_sec[0], "c_min_entry", config_vals[5])
    gui_config.set(entry_sec[0], "c_max_entry", config_vals[6])
    gui_config.set(entry_sec[0], "density_status", config_vals[7])
    gui_config.set(entry_sec[0], "norm_type", config_vals[8])
    gui_config.set(entry_sec[0], "unit_type", config_vals[9])
    gui_config.set(entry_sec[0], "v_min_thresh_entry", config_vals[10])
    gui_config.set(entry_sec[0], "v_max_thresh_entry", config_vals[11])
    gui_config.set(entry_sec[0], "v_sum_min_thresh_entry", config_vals[12])
    gui_config.set(entry_sec[0], "v_sum_max_thresh_entry", config_vals[13])
    gui_config.set(entry_sec[0], "cut_status", config_vals[14])
    gui_config.set(entry_sec[0], "curve_fit_status", config_vals[15])
    gui_config.set(entry_sec[0], "lin_corr_status", config_vals[16])
    gui_config.set(entry_sec[0], "cmap", config_vals[17])

    gui_config.add_section(entry_sec[1])
    gui_config.set(entry_sec[1], "start_time", config_vals[18])
    gui_config.set(entry_sec[1], "end_time", config_vals[19])
    with open("luigi.cfg", "w") as config_file:
        gui_config.write(config_file)
    print("\033[1;32mConfiguration file 'luigi.cfg' updated.\033[0m")
