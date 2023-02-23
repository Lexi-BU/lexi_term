import glob
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from spacepy.pycdf import CDF as cdf

import lxi_file_read_funcs as lxrf

importlib.reload(lxrf)


def lxi_csv_to_cdf(df=None, csv_file=None, csv_folder=None, cdf_file=None, cdf_folder=None):
    """
    Convert a CSV file to a CDF file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.
    csv_folder : str
        Path to the folder containing the CSV files.
    cdf_file : str
        Path to the CDF file.
    cdf_folder : str
        Path to the folder containing the CDF files.

    Returns
    -------
    cdf_file : str
        Path to the CDF file.
    """

    if csv_folder is not None:
        if csv_file is not None:
            csv_file_list = [csv_folder + "/" + csv_file]
        else:
            # Find all the CSV files in the folder
            csv_file_list = np.sort(glob.glob(csv_folder + "*.csv"))
    elif csv_folder is None and csv_file is not None:
        csv_file_list = [csv_file]

    for csv_file in csv_file_list:
        if df is None:
            df, _ = lxrf.read_csv_sci(csv_file)
            csv_file_secs = int(csv_file.split("_")[-4])
            csv_file_subsecs = int(csv_file.split("_")[-3])
            # Create a datetime array for the CSV file with datetime objects as type
            csv_file_datetime = np.full(len(df.index), np.nan)
            for xx, time_ind in enumerate(df.index):
                csv_file_datetime[xx] = csv_file_secs + csv_file_subsecs / 1e3 + time_ind
            df.index = csv_file_datetime
        else:
            df = df
            csv_file_secs = int(csv_file.split("_")[-4])
            csv_file_subsecs = int(csv_file.split("_")[-3])
            # Create a datetime array for the CSV file with datetime objects as type
            csv_file_datetime = np.full(len(df.index), dtype=object, fill_value=np.nan)
            for xx, time_ind in enumerate(df.index):
                csv_file_datetime[xx] = csv_file_secs + csv_file_subsecs / 1e3 + time_ind
            df.index = csv_file_datetime

        # If the cdf_folder does not exist, create it
        if cdf_folder is not None:
            if not Path(cdf_folder).exists():
                Path(cdf_folder).mkdir(parents=True, exist_ok=True)
                print(f"\n \033[1;32mCreated folder {cdf_folder}\033[0m \n")
        else:
            cdf_folder = "/".join(csv_file.split("/")[0:-2]) + "/cdf"
            if not Path(cdf_folder).exists():
                Path(cdf_folder).mkdir(parents=True, exist_ok=True)
                print(f"\n \033[1;32mCreated folder {cdf_folder}\033[0m \n")

        # If the cdf_file does not exist, create it
        if cdf_file is None:
            cdf_file = cdf_folder + "/" + csv_file.split("/")[-1].split(".")[0] + ".cdf"
        else:
            cdf_file = cdf_folder + "/" + cdf_file
            print("Creating CDF file: " + cdf_file)

        # If the cdf file already exists, overwrite it
        if Path(cdf_file).exists():
            # Raise a warning saying the file already exists and ask the user if they want to
            # overwrite it
            print(f"\n \x1b[1;31;255m WARNING: The CDF file already exists and will be overwritten:"
                  f" {cdf_file} \x1b[0m")
            Path(cdf_file).unlink()

        print(f"Creating CDF file: {cdf_file}")
        cdf_data = cdf(cdf_file, "")
        cdf_data.attrs["title"] = csv_file.split("/")[-1].split(".")[0]
        cdf_data.attrs["created"] = str(pd.Timestamp.now())
        cdf_data.attrs["TimeZone"] = "UTC"
        cdf_data.attrs["creator"] = "Ramiz A. Qudsi"
        cdf_data.attrs["source"] = csv_file
        cdf_data.attrs["source_type"] = "csv"
        cdf_data.attrs["source_format"] = "lxi"
        cdf_data.attrs["source_version"] = "0.1.0"
        cdf_data.attrs["source_description"] = "LXI data from the LXI data logger"
        cdf_data.attrs["source_description_url"] = "something"
        cdf_data.attrs["source_description_email"] = "qudsira@bu.edu"
        cdf_data.attrs["source_description_institution"] = "BU"

        # Convert the array to datetime objects in UTC
        df.index = pd.to_datetime(df.index, utc=True, unit="s")
        cdf_data["Epoch"] = df.index
        for col in df.columns:
            cdf_data[col] = df[col]
        cdf_data.close()
        print(f"\n  CDF file created: \x1b[1;32;255m {cdf_file} \x1b[0m")

    return cdf_file
