import importlib

import lxi_file_read_funcs as lxrf

# Read a binary data file
"""
# Science dataset
df_sci, save_file_name_sci = lxrf.read_binary_data_sci(
    in_file_name='../data/sample_binary_dataset.txt',
    save_file_name='dataset_sci.csv',
    number_of_decimals=3)

# Housekeeping dataset
df_hk, save_file_name_hk = lxrf.read_binary_data_hk(
    in_file_name='../data/sample_binary_dataset.txt',
    save_file_name='dataset_hk.csv',
    number_of_decimals=3)
"""
# Get the science and housekeeping dataframes with corrected positions and voltages
df_slice_hk, file_name_hk, df_slice_sci, file_name_sci, df_hk, df_sci = lxrf.read_binary_file(
    file_val='../data/20221114/payload_lexi_1706245848_39194.dat', t_start=None, t_end=None
    )
