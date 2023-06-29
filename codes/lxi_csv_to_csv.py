import importlib
from pathlib import Path

import lxi_file_read_funcs as lxrf

importlib.reload(lxrf)


def lxi_csv_to_csv(
    df=None, csv_file=None, csv_folder=None, csvs_file=None, csvs_folder=None
):
    """
    Convert a CSV file to a CDF file.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV file.
    csv_folder : str
        Path to the folder containing the input CSV files.
    csvs_file : str
        Path to the output CSV file.
    csvs_folder : str
        Path to the folder containing the output CSV files.

    Returns
    -------
    csvs_file : str
        Path to the output CSV file.
    """

    # NOTE: This part is for future when we want to convert multiple CSV files to a single CSV file
    # if csv_folder is not None:
    #     if csv_file is not None:
    #         csv_file_list = [csv_folder + "/" + csv_file]
    #     else:
    #         # Find all the CSV files in the folder
    #         csv_file_list = np.sort(glob.glob(csv_folder + "*.csv"))
    # elif csv_folder is None and csv_file is not None:
    #     csv_file_list = [csv_file]

    # NOTE: This loop was implemented so that CDF creation would work for the GSFC file. However,
    # given that the PIT files have time stamps correctly ordered in thier indices, this is no
    # longer necessary and thus has been commented out.
    # for csv_file in csv_file_list:
    #     if df is None:
    #         df, _ = lxrf.read_csv_sci(csv_file)
    #         csv_file_secs = int(csv_file.split("_")[-4])
    #         csv_file_subsecs = int(csv_file.split("_")[-3])
    #         # Create a datetime array for the CSV file with datetime objects as type
    #         csv_file_datetime = np.full(len(df.index), np.nan)
    #         for xx, time_ind in enumerate(df.index):
    #             csv_file_datetime[xx] = csv_file_secs + csv_file_subsecs / 1e3 + time_ind
    #         df.index = csv_file_datetime
    #     else:
    #         df = df
    #         csv_file_secs = int(csv_file.split("_")[-4])
    #         csv_file_subsecs = int(csv_file.split("_")[-3])
    #         # Create a datetime array for the CSV file with datetime objects as type
    #         csv_file_datetime = np.full(len(df.index), dtype=object, fill_value=np.nan)
    #         for xx, time_ind in enumerate(df.index):
    #             csv_file_datetime[xx] = csv_file_secs + csv_file_subsecs / 1e3 + time_ind
    #         df.index = csv_file_datetime

    # If the csvs_folder does not exist, create it
    if csvs_folder is not None:
        if not Path(csvs_folder).exists():
            Path(csvs_folder).mkdir(parents=True, exist_ok=True)
            print(f"\n \033[1;32mCreated folder {csvs_folder}\033[0m \n")
    else:
        csvs_folder = "/".join(csv_file.split("/")[0:-2]) + "/csv"
        if not Path(csvs_folder).exists():
            Path(csvs_folder).mkdir(parents=True, exist_ok=True)
            print(f"\n \033[1;32mCreated folder {csvs_folder}\033[0m \n")

        # If the csvs_file does not exist, create it
        if csvs_file is None:
            csvs_file = (
                csvs_folder + "/" + csv_file.split("/")[-1].split(".")[0] + ".csv"
            )
        else:
            csvs_file = csvs_folder + "/" + csvs_file
            print("Creating csv file: " + csvs_file)

        # If the csv file already exists, overwrite it
        if Path(csvs_file).exists():
            # Raise a warning saying the file already exists and ask the user if they want to
            # overwrite it
            print(
                f"\n \x1b[1;31;255m WARNING: The csv file already exists and will be overwritten:"
                f" {csvs_file} \x1b[0m"
            )
            Path(csvs_file).unlink()

        print(f"Creating csv file: {csvs_file}")

        # Save to data to a CSV file
        df.to_csv(csvs_file, index=True, header=True)

        print(f"\n  CSV file created: \x1b[1;32;255m {csvs_file} \x1b[0m")

    return csvs_file
