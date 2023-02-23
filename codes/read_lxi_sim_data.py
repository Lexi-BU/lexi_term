import importlib
import numpy as np
import pandas as pd
import glob

import lxi_file_read_funcs as lxrf

importlib.reload(lxrf)

folder_name = "../data/raw_data/lxi_sim_data/PIT/archives/files/downloads/20220804/"
file_list = np.sort(glob.glob(f"{folder_name}*.dat"))

save_folder_sci = f"{folder_name[:-9]}/processed_data/sci/"
save_folder_hk = f"{folder_name[:-9]}/processed_data/hk/"

for file in file_list[0:]:
    df_sci, fn = lxrf.read_binary_data_sci(in_file_name=file, save_file_name=save_folder_sci)

    df_hk, fn = lxrf.read_binary_data_hk(in_file_name=file, save_file_name=save_folder_hk)

sci_csv_file_list = np.sort(glob.glob(f"{save_folder_sci}*.csv"))
hk_csv_file_list = np.sort(glob.glob(f"{save_folder_hk}*.csv"))

# Combine all the CSV files into one
df_sci = pd.concat([pd.read_csv(f) for f in sci_csv_file_list])
df_hk = pd.concat([pd.read_csv(f) for f in hk_csv_file_list])

# Save the combined CSV file
df_sci.to_csv(f"{save_folder_sci}combined_sci.csv", index=False)
df_hk.to_csv(f"{save_folder_hk}combined_hk.csv", index=False)
