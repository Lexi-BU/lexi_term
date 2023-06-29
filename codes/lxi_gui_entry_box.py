import importlib
import tkinter as tk
from tkinter import font, ttk
import lxi_gui_config as lgcf

importlib.reload(lgcf)


def entry_box(
    root=None,
    width=10,
    justify="center",
    bg="black",
    fg="white",
    borderwidth=2,
    entry_val="",
    entry_label="",
    row=0,
    column=0,
    rowspan=1,
    columnspan=1,
    sticky="n",
    font_style=None,
):
    """
    Creates a text entry box in the GUI.

    Parameters
    ----------
    root : tkinter.Tk
        The root window of the GUI.
    width : int
        The width of the entry box. Default is 10.
    justify : str
        The justification of the entry box. Default is "center".
    bg : str
        The background color of the entry box. Default is "white".
    fg : str
        The foreground color of the entry box. Default is "black".
    borderwidth : int
        The border width of the entry box. Default is 2.
    entry_val : str
        The initial value of the entry box. Default is "".
    entry_label : str
        The label of the entry box. Default is "".
    row : int
        The row in which the entry box should be displayed. Default is 0.
    column : int
        The column in which the entry box should be displayed. Default is 0.
    rowspan : int
        The number of rows that the entry box should span. Default is 1.
    columnspan : int
        The number of columns that the entry box should span. Default is 1.
    sticky : str
        The sticky parameter for the grid. Default is "n".
    font_style : str
        The font style of the entry box. Default is None.

    Raises
    ------
    ValueError:f the root is not specified.

    Returns
    -------
    main_entry : tkinter.Entry
        The entry box created in the GUI.
    main_label : tkinter.Label
        The label created in the GUI.
    """

    # Check if row and column has length attribute
    # if not hasattr(row, "__len__"):
    #     row = [row] * 2
    # if not hasattr(column, "__len__"):
    #     column = [column] * 2

    if root is None:
        raise ValueError("Root is not defined.")
    if font_style is None:
        font_style = font.Font(family="Helvetica", size=12)

    main_entry = tk.Entry(root, width=width, justify=justify, bg=bg, fg=fg, borderwidth=borderwidth)  # type: ignore
    main_entry.insert(0, entry_val)
    main_entry.grid(
        row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky=sticky
    )
    main_label = tk.Label(root, text=entry_label, font=font_style, bg=bg, fg=fg)
    main_label.grid(
        row=row,
        column=column + 1,
        columnspan=columnspan,
        rowspan=rowspan,
        sticky=sticky,
    )

    return main_entry


def populate_entries(root=None, dark_mode=True, default_vals=False):
    """
    Populates the entry boxes in the GUI for the Sci Tab.

    Parameters
    ----------
    root : tkinter.Tk
        The root window of the GUI.
    dark_mode : bool
        If True, the GUI is in dark mode. Default is True.

    Raises
    ------
    ValueError : If the root is not specified.

    Returns
    -------
    None.
    """

    if root is None:
        raise ValueError("Root is not specified.")

    if dark_mode:
        bg_color = "black"
        fg_color = "white"
    else:
        bg_color = "white"
        fg_color = "black"
    # Get the default values from the config file
    default_opt_dict = lgcf.get_config_entry(default_vals=default_vals)

    # The minimum value of x-axis for histogram plot
    opt_col_num = 9
    opt_col_num2 = 10

    font_style_box = font.Font(family="serif", size=12, weight="bold")

    # The minimum value of x-axis for the histogram plot
    x_min_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=0,
        column=opt_col_num,
        entry_label="X-min",
        entry_val=default_opt_dict["x_min_entry"],
        font_style=font_style_box,
    )

    # The maximum value of x-axis for histogram plot
    x_max_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=1,
        column=opt_col_num,
        entry_label="X-max",
        entry_val=default_opt_dict["x_max_entry"],
        font_style=font_style_box,
    )
    # The minimum value of y-axis for histogram plot
    y_min_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=2,
        column=opt_col_num,
        entry_label="Y-min",
        entry_val=default_opt_dict["y_min_entry"],
        font_style=font_style_box,
    )

    # The maximum value of y-axis for histogram plot
    y_max_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=3,
        column=opt_col_num,
        entry_label="Y-max",
        entry_val=default_opt_dict["y_max_entry"],
        font_style=font_style_box,
    )

    # The number of bins for histogram plot
    hist_bins_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=4,
        column=opt_col_num,
        entry_label="Bins",
        entry_val=default_opt_dict["hist_bin_entry"],
        font_style=font_style_box,
    )

    # Mimimum number of data points in each bin for the histogram plot
    c_min_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=5,
        column=opt_col_num,
        entry_label="C Min",
        entry_val=default_opt_dict["c_min_entry"],
        font_style=font_style_box,
    )

    # Maximum number of data points in each bin for the histogram plot
    c_max_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=6,
        column=opt_col_num,
        entry_label="C Max",
        entry_val=default_opt_dict["c_max_entry"],
        font_style=font_style_box,
    )

    # Choose whether to plot probability density or the number of data points in each bin (is Bool)
    density_label = tk.Label(root, text="Density", font=font_style_box, bg=bg_color, fg=fg_color)
    density_label.grid(row=7, column=opt_col_num2, columnspan=1, sticky="n")

    # Add a checkbox to choose whether to plot probability density or the number of data points in
    # each bin
    density_status_var = tk.BooleanVar()
    density_status_var.set(default_opt_dict["density_status"])  # type: ignore
    density_checkbox = tk.Checkbutton(
        root,
        bg=bg_color,
        fg=fg_color,
        text="",
        font=font_style_box,
        variable=density_status_var,
        highlightcolor=bg_color,
        selectcolor="#808080",
        cursor="hand2",
    )
    density_checkbox.grid(row=7, column=opt_col_num, columnspan=1, sticky="n")

    # Key for the norm of the colorbar
    norm_label = tk.Label(
        root, text="Norm", font=font_style_box, bg=bg_color, fg=fg_color
    )
    norm_label.grid(row=8, column=opt_col_num2, columnspan=1, sticky="n")

    # Add dropdown menu for the norm type (options are log or linear)
    norm_type_var = tk.StringVar()
    norm_type_var.set(default_opt_dict["norm_type"])
    norm_type_combobox = ttk.Combobox(
        root, textvariable=norm_type_var, values=["log", "linear"], state="readonly"
    )

    norm_type_combobox.grid(row=8, column=opt_col_num, columnspan=1, sticky="new")

    # Add a dropdown menu for the unit type (options are volt, mcp, deg)
    unit_label = tk.Label(root, text="Unit", font=font_style_box, bg=bg_color, fg=fg_color)
    unit_label.grid(row=10, column=opt_col_num2, columnspan=1, sticky="n")

    unit_type_var = tk.StringVar()
    unit_type_var.set(default_opt_dict["unit_type"])

    unit_type_combobox = ttk.Combobox(
        root, textvariable=unit_type_var, values=["volt", "mcp", "deg"], state="readonly"
    )
    unit_type_combobox.grid(row=10, column=opt_col_num, columnspan=1, sticky="new")

    # Minimum threshold for the voltage to be considered
    v_min_thresh_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=12,
        column=opt_col_num,
        entry_label="V Min",
        entry_val=default_opt_dict["v_min_thresh_entry"],
        font_style=font_style_box,
    )

    # Maximum threshold for the voltage to be considered
    v_max_thresh_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=13,
        column=opt_col_num,
        entry_label="V Max",
        entry_val=default_opt_dict["v_max_thresh_entry"],
        font_style=font_style_box,
    )

    # Sum of minimum and maximum threshold for the voltage to be considered
    v_sum_min_thresh_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=14,
        column=opt_col_num,
        entry_label="V sum Min",
        entry_val=default_opt_dict["v_sum_min_thresh_entry"],
        font_style=font_style_box,
    )

    v_sum_max_thresh_entry = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=15,
        column=opt_col_num,
        entry_label="V sum Max",
        entry_val=default_opt_dict["v_sum_max_thresh_entry"],
        font_style=font_style_box,
    )

    # Choose whether to plot the horizontal and vertical lines on the hist plot
    cut_v_labels = tk.Label(
        root, text="crosswire", font=font_style_box, bg=bg_color, fg=fg_color
    )
    cut_v_labels.grid(row=16, column=opt_col_num2, columnspan=1, sticky="n")

    # Add a checkbox to choose whether to plot the vertical and horizontal cuts or not
    cut_status_var = tk.BooleanVar()
    cut_status_var.set(default_opt_dict["cut_status"])  # type: ignore
    cut_checkbox = tk.Checkbutton(
        root,
        bg=bg_color,
        fg=fg_color,
        text="",
        font=font_style_box,
        variable=cut_status_var,
        highlightcolor=bg_color,
        selectcolor="#808080",
        cursor="hand2",
    )
    cut_checkbox.grid(row=16, column=opt_col_num, columnspan=1, sticky="n")

    # Choose whether to plot curve fit or not (is Bool)
    curve_fit_label = tk.Label(
        root, text="Curve Fit", font=font_style_box, bg=bg_color, fg=fg_color
    )
    curve_fit_label.grid(row=17, column=opt_col_num2, columnspan=1, sticky="n")

    # Add a checkbox to choose whether to plot curve fit or not
    curve_fit_status_var = tk.BooleanVar()
    curve_fit_status_var.set(default_opt_dict["curve_fit_status"])  # type: ignore
    curve_fit_checkbox = tk.Checkbutton(
        root,
        bg=bg_color,
        fg=fg_color,
        text="",
        font=font_style_box,
        variable=curve_fit_status_var,
        highlightcolor=bg_color,
        selectcolor="#808080",
        cursor="hand2",
    )
    curve_fit_checkbox.grid(row=17, column=opt_col_num, columnspan=1, sticky="n")

    # Choose whether to implement linear correction or not (is Bool)
    lin_corr_label = tk.Label(
        root, text="Correction", font=font_style_box, bg=bg_color, fg=fg_color
    )
    lin_corr_label.grid(row=18, column=opt_col_num2, columnspan=1, sticky="n")

    # Add a checkbox to choose whether to implement non-linearity correction or not
    lin_corr_status_var = tk.BooleanVar()
    lin_corr_status_var.set(default_opt_dict["lin_corr_status"])  # type: ignore
    lin_corr_checkbox = tk.Checkbutton(
        root,
        bg=bg_color,
        fg=fg_color,
        text="",
        font=font_style_box,
        variable=lin_corr_status_var,
        highlightcolor="green",
        selectcolor="#808080",
        cursor="hand2",
    )
    lin_corr_checkbox.grid(row=18, column=opt_col_num, columnspan=1, sticky="n")

    # Choose the colorbar
    cmap_option = entry_box(
        root=root,
        bg=bg_color,
        fg=fg_color,
        row=19,
        column=opt_col_num,
        entry_label="Cmap",
        entry_val=default_opt_dict["cmap"],
        font_style=font_style_box,
    )

    return (
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
    )
