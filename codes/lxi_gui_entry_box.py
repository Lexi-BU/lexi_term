import tkinter as tk
from tkinter import font


def entry_box(root=None,
              width=10,
              justify="center",
              bg="white",
              fg="black",
              borderwidth=2,
              entry_val="",
              entry_label="",
              row=0,
              column=0,
              rowspan=1,
              columnspan=1,
              sticky="n",
              font_style=None):
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

    main_entry = tk.Entry(root, width=width, justify=justify, bg=bg, fg=fg, borderwidth=borderwidth)
    main_entry.insert(0, entry_val)
    main_entry.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky=sticky)
    main_label = tk.Label(root, text=entry_label, font=font_style, bg=bg, fg=fg)
    main_label.grid(row=row, column=column + 1, columnspan=columnspan, rowspan=rowspan,
                    sticky=sticky)

    return main_entry
