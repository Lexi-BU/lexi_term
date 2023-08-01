import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from multiprocessing import Pool

import time


plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def vignette(d):
    f = 1
    return f


def interpolate_pointing(
    df,
    res="100L",
    method="ffill",
    save_df=False,
    filename="../data/LEXI_pointing_ephem_highres",
    filetype="pkl",
):
    # Check if df is a pandas dataframe and is not empty
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas dataframe")
    if df.empty:
        raise ValueError("df cannot be empty")
    if not isinstance(res, str):
        raise TypeError(
            "res must be a string. Options are '100L' or '1S' or any other pandas"
            "resample string"
        )

    # Check if the index is a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index must be a datetime object")

    if method not in ["ffill", "bfill"]:
        raise ValueError("method must be 'ffill' or 'bfill'")

    # Interpolate the even time steps to get 100 ms resolution
    # Print that the interpolation is happening
    print(
        f"\nInterpolating the pointing data to {res} resolution. This may take a while\n"
    )
    df_intrp = df.resample(res).interpolate(method=method)

    if save_df:
        print(f"Trying to save dataframe to {filename}_res_{res}.{filetype}\n")
        # Check if filename is given, if not then print saying that default will be used
        if filename is None:
            print(f"No filename given, default will be used: {filename}")
        # Check if filename is a string
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        # Save the dataframe to proper file type
        if filetype == "csv":
            df_intrp.to_csv(f"{filename}_res_{res}.{filetype}", index=False)
        elif filetype == "pkl":
            df_intrp.to_pickle(f"{filename}_res_{res}.{filetype}")
        else:
            raise ValueError("filetype must be 'csv' or 'pkl'")
        print(f"Dataframe saved to {filename}_res_{res}.{filetype}\n")

    return df_intrp


def parallelize_exposure(
    df,
    x_grid,
    y_grid,
    x_range,
    y_range,
    LEXI_FOV,
    roll,
    step,
    num_cores=4,
):
    # Split the dataframe into chunks based on the number of cores
    df_split = np.array_split(df, num_cores)

    # Create a pool of workers
    with Pool(num_cores) as pool:
        # Calculate the exposure for each chunk in parallel
        results = pool.map(
            lambda df_chunk: calculate_exposure(
                df_chunk, x_grid, y_grid, x_range, y_range, LEXI_FOV, roll, step
            ),
            df_split,
        )

    # Combine the results from the individual chunks
    exposure = np.concatenate(results)

    return exposure


def calculate_exposure(
    df_chunk,
    x_grid,
    y_grid,
    x_range,
    y_range,
    LEXI_FOV,
    roll,
    step,
):
    # Initialize the exposure map
    exposure = np.zeros_like(x_grid)

    # Loop through each pointing step in the chunk and add the exposure to the map
    for i in range(len(df_chunk)):
        r = np.sqrt(
            (x_grid - df_chunk.mp_ra[i]) ** 2 + (y_grid - df_chunk.mp_dec[i]) ** 2
        )  # Get distance in degrees to the pointing step
        exposure_delt = np.where(
            (r < LEXI_FOV * 0.5), vignette(r) * step, 0
        )  # Make an exposure delta for this span
        exposure += exposure_delt  # Add the delta to the full map

    return exposure


if __name__ == "__main__":
    # Set the input parameters
    input_dict = {
        "res": "1000L",  # Time resolution to interpolate to.  Default is 100 ms
        "method": "ffill",  # Interpolation method.  Default is forward fill
        "save_df": True,  # If True, save the dataframe to a file.  Default is False
        "filename": "../data/LEXI_pointing_ephem_highres",  # Filename to save the dataframe to.  Default is '../data/LEXI_pointing_ephem_highres'
        "filetype": "pkl",  # Filetype to save the dataframe to.  Default is 'pkl'. Options are 'csv' or 'pkl'
        "x_res": 1,  # x res in degrees. Ideal value is 0.1 deg
        "y_res": 1,  # y res in degrees. Ideal value is 0.1 deg
        "LEXI_FOV": 9.1,  # LEXI FOV in degrees
        "roll": 0.0,  # deg roll angle.  Here 0 deg will correspond to line up perfectly with RA/DEC
        "xrange": [325.0, 365.0],  # desired input for plotting ranges in RA
        "yrange": [-21.0, 6.0],  # desired input for plotting ranges in DEC
        "x_offset": 0.0,  # deg angle from Az of mounting plate value to RA
        "y_offset": 9.1 / 2.0,  # deg angle from El of mounting plate value to DEC
        "step": 0.01,  # step size in seconds
    }

    # Load the pointing data
    df = pd.read_pickle(
        f"../data/LEXI_pointing_ephem_highres_res_{input_dict['res']}.pkl"
    )

    # Set up the exposure map
    x_grid = np.arange(
        input_dict["xrange"][0], input_dict["xrange"][1], input_dict["x_res"]
    )
    y_grid = np.arange(
        input_dict["yrange"][0], input_dict["yrange"][1], input_dict["y_res"]
    )

    # Calculate the exposure map in parallel
    exposure = parallelize_exposure(
        df,
        x_grid,
        y_grid,
        input_dict["xrange"],
        input_dict["yrange"],
        input_dict["LEXI_FOV"],
        input_dict["roll"],
        input_dict["step"],
    )

    # Save the exposure map
    np.save(
        f"../data/exposure_map_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_v12.npy",
        exposure,
    )
