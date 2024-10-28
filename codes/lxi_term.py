import importlib
from pathlib import Path

import global_variables
import lxi_file_read_funcs as lxrf
import lxi_gui_plot_routines as lgpr

importlib.reload(lxrf)
importlib.reload(lgpr)
importlib.reload(global_variables)

global_variables.init()


# Write a function that takes filename and dataframe as input and saves the dataframe to a csv file
def save_df_to_csv(
    file_name,
    df,
    threshold=False,
    v_min=1.2,
    v_max=3.4,
    v_sum_min=5.0,
    v_sum_max=7,
    number_of_decimals=3,
):
    """
    Save the dataframe to a csv file

    Parameters
    ----------
    file_name : str
        Name of the file to be saved
    df : pandas.DataFrame
        Dataframe to be saved
    threshold : bool
        If True, apply threshold to the dataframe. Default is False
    v_min : float
        Minimum voltage value corresponding to threshold. Default is 1.2
    v_max : float
        Maximum voltage value corresponding to threshold. Default is 3.4
    v_sum_min : float
        Minimum sum of voltage values corresponding to threshold. Default is 5.0
    v_sum_max : float
        Maximum sum of voltage values corresponding to threshold. Default is 7.0
    number_of_decimals : int
        Number of decimal places to round the dataframe to. Default is 3

    Returns
    -------
    None

    """
    # Create a new file name
    # Expand the file name to include the updated keyword
    file_name = Path(file_name).expanduser()
    new_file_name = file_name.parent / (file_name.stem + "_updated" + file_name.suffix)

    print("Saving dataframe to file: ", new_file_name)
    if threshold:
        print("Applying threshold to the dataframe")
        df = df[
            (df["Channel1"] >= v_min)
            & (df["Channel1"] <= v_max)
            & (df["Channel2"] >= v_min)
            & (df["Channel2"] <= v_max)
            & (df["Channel3"] >= v_min)
            & (df["Channel3"] <= v_max)
            & (df["Channel4"] >= v_min)
            & (df["Channel4"] <= v_max)
            & (
                (df["Channel1"] + df["Channel2"] + df["Channel3"] + df["Channel4"])
                >= v_sum_min
            )
            & (
                (df["Channel1"] + df["Channel2"] + df["Channel3"] + df["Channel4"])
                <= v_sum_max
            )
        ]

    # Save the dataframe to a csv file rounded to 3 decimal places
    df.round(number_of_decimals).to_csv(new_file_name, index=True)
    print("Done saving dataframe to file: ", new_file_name)


# Read a binary data file
# Get the science and housekeeping dataframes with corrected positions and voltages

file_val = "/home/cephadrius/Desktop/git/Lexi-BU/lexi_term/data/GSFC/2022_04_21_1431_LEXI_HK_unit_1_mcp_unit_1_eBox_1987_hk_/2022_04_21_1431_LEXI_raw_LEXI_unit_1_mcp_unit_1_eBox-1987.txt"
(
    df_slice_hk,
    file_name_hk,
    df_slice_sci,
    file_name_sci,
    df_hk,
    df_sci,
) = lxrf.read_binary_file(
    # file_val="../data/PIT/20221114/payload_lexi_1706245848_39194.dat",
    file_val=file_val,
    t_start=None,
    t_end=None,
)

# Save the science dataframe to a csv file
save_df_to_csv(file_name_sci, df_sci, threshold=False)
save_df_to_csv(file_name_hk, df_hk, threshold=False)

# Get the figure name using the file_val
fig_name = file_val.split("/")[-1].split(".")[0]

# Plot the histogram of the science data

fig_hist = lgpr.plot_data_class(
    df_slice_sci=df_slice_sci,
    # start_time=start_time,
    # end_time=end_time,
    bins=50,
    cmin=0,
    cmax=100,
    x_min=-5,
    x_max=5,
    y_min=-5,
    y_max=5,
    density=False,
    norm="linear",
    unit="mcp",
    hist_fig_height=10,
    hist_fig_width=10,
    v_min=0,
    v_max=5,
    v_sum_min=0,
    v_sum_max=20,
    cut_status_var=False,
    crv_fit=False,
    lin_corr=True,
    non_lin_corr=True,
    cmap="viridis",
    # use_fig_size=use_fig_size,
    dark_mode=False,
    save_file_name=fig_name,
).hist_plots()
