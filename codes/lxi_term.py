import importlib

import lxi_file_read_funcs as lxrf

importlib.reload(lxrf)


# Write a function that takes filename and dataframe as input and saves the dataframe to a csv file
def save_df_to_csv(file_name, df, threshold=False, v_min=1.2, v_max=3.4, v_sum_min=5.0,
                   v_sum_max=7, number_of_decimals=3):
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
    new_file_name = ".." + file_name.split('.')[2] + "_updated" + ".csv"

    print("Saving dataframe to file: ", new_file_name)
    if threshold:
        print("Applying threshold to the dataframe")
        df = df[(df["Channel1"] >= v_min) & (df["Channel1"] <= v_max) &
                (df["Channel2"] >= v_min) & (df["Channel2"] <= v_max) &
                (df["Channel3"] >= v_min) & (df["Channel3"] <= v_max) &
                (df["Channel4"] >= v_min) & (df["Channel4"] <= v_max) &
                ((df["Channel1"] + df["Channel2"] + df["Channel3"] + df["Channel4"]) >= v_sum_min) &
                ((df["Channel1"] + df["Channel2"] + df["Channel3"] + df["Channel4"]) <= v_sum_max)]

    # Save the dataframe to a csv file rounded to 3 decimal places
    df.round(number_of_decimals).to_csv(new_file_name, index=True)
    print("Done saving dataframe to file: ", new_file_name)


# Read a binary data file
# Get the science and housekeeping dataframes with corrected positions and voltages
df_slice_hk, file_name_hk, df_slice_sci, file_name_sci, df_hk, df_sci = lxrf.read_binary_file(
    file_val="../data/PIT/20221114/payload_lexi_1706245848_39194.dat", t_start=None, t_end=None
)

# df_slice_hk, file_name_hk, df_slice_sci, file_name_sci, df_hk, df_sci = lxrf.read_binary_file(
#     file_val="../data/GSFC/2022_04_21_1431_LEXI_HK_unit_1_mcp_unit_1_eBox_1987_hk_/"
#              "2022_04_21_1431_LEXI_raw_LEXI_unit_1_mcp_unit_1_eBox-1987.txt", t_start=None,
#              t_end=None
#     )


# Save the science dataframe to a csv file
save_df_to_csv(file_name_sci, df_sci, threshold=False)
save_df_to_csv(file_name_hk, df_hk, threshold=False)
