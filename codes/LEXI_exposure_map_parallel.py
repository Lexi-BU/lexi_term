import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import time


# Set latex use to true
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# --------------- Run things inside function ---------------
# -------- input parts to run mask ----------------


def vignette(d):
    """
    Function to calculate the vignetting factor for a given distance from boresight

    Parameters
    ----------
    d : float
        Distance from boresight in degrees

    Returns
    -------
    f : float
        Vignetting factor
    """

    # Set the vignetting factor
    # f = 1.0 - 0.5 * (d / (LEXI_FOV * 0.5)) ** 2
    f = 1

    return f


def calculate_exposure_delta(*args):
    """
    Function to calculate the exposure delta for a given pointing step

    Parameters
    ----------
    args : tuple
        Tuple of arguments to pass to the function
        The arguments are:
            i : int
                Index of the pointing step
            x_grid_arr : numpy array
                Array of x values for the exposure map
            y_grid_arr : numpy array
                Array of y values for the exposure map
            ra : float
                RA of the pointing step
            dec : float
                DEC of the pointing step
            LEXI_FOV : float
                LEXI FOV in degrees
            step : float
                Step size in seconds

    Returns
    -------
    exposure_delt : numpy array
        Exposure delta array
    """
    i = args[0][0]
    x_grid_arr = args[0][1]
    y_grid_arr = args[0][2]
    ra = args[0][3]
    dec = args[0][4]
    LEXI_FOV = args[0][5]
    step = args[0][6]

    # Calculate the distance from the pointing to each pixel
    r = np.sqrt((x_grid_arr - ra) ** 2 + (y_grid_arr - dec) ** 2)
    exposure_delt = np.where((r < LEXI_FOV * 0.5), vignette(r) * step, 0)
    return exposure_delt


# Function to interpolate the pointing data to a given resolution
def interpolate_pointing(
    df,
    res="100L",
    method="ffill",
    save_df=False,
    filename="../data/LEXI_pointing_ephem_highres",
    filetype="pkl",
):
    """
    Function to interpolate the pointing data to a given resolution

    Parameters
    ----------
    df : pandas dataframe
        Dataframe with pointing data
    res : string
        Resolution to interpolate to.  Default is 100 ms
    method : string
        Interpolation method.  Default is forward fill
    save_df : bool
        If True, save the dataframe to a file.  Default is False
    filename : string
        Filename to save the dataframe to.  Default is '../data/LEXI_pointing_ephem_highres'
    filetype : string
        Filetype to save the dataframe to.  Default is 'pkl'. Options are 'csv' or 'pkl'

    Returns
    -------
    df : pandas dataframe
        Dataframe with interpolated pointing data

    Raises
    ------
    TypeError
        If df is not a pandas dataframe
    ValueError
        If df is empty
    TypeError
        If res is not a string
    TypeError
        If index is not a datetime object
    ValueError
        If method is not 'ffill' or 'bfill'
    TypeError
        If filename is not a string
    """

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


code_start_time = time.time()

input_dict = {
    "res": "100L",  # Time resolution to interpolate to.  Default is 100 ms
    "method": "ffill",  # Interpolation method.  Default is forward fill
    "save_df": True,  # If True, save the dataframe to a file.  Default is False
    "filename": "../data/LEXI_pointing_ephem_highres",  # Filename to save the dataframe to.  Default is '../data/LEXI_pointing_ephem_highres'
    "filetype": "pkl",  # Filetype to save the dataframe to.  Default is 'pkl'. Options are 'csv' or 'pkl'
    "x_res": 0.1,  # x res in degrees. Ideal value is 0.1 deg
    "y_res": 0.1,  # y res in degrees. Ideal value is 0.1 deg
    "LEXI_FOV": 9.1,  # LEXI FOV in degrees
    "roll": 0.0,  # deg roll angle.  Here 0 deg will correspond to line up perfectly with RA/DEC
    "xrange": [325.0, 365.0],  # desired input for plotting ranges in RA
    "yrange": [-21.0, 6.0],  # desired input for plotting ranges in DEC
    "x_offset": 0.0,  # deg angle from Az of mounting plate value to RA
    "y_offset": 9.1 / 2.0,  # deg angle from El of mounting plate value to DEC
    "step": 0.01,  # step size in seconds
    "n_cores": 2,  # number of cores to use
}

# Try to read in the high res pointing file, if it doesn't exist then make it
try:
    df = pd.read_pickle(
        f"../data/LEXI_pointing_ephem_highres_res_{input_dict['res']}.pkl"
    )
    print("\n High res pointing file loaded from file \n")
except FileNotFoundError:
    print("High res pointing file not found, computing now. This may take a while \n")
    ephem = pd.read_csv("SAMPLE_LEXI_pointing_ephem_edited.csv", sep=",")

    # Set 'epoch_utc' column to datetime object, and set time to UTC
    ephem["epoch_utc"] = pd.to_datetime(ephem["epoch_utc"], utc=True)

    # Set index to 'epoch_utc' column, also keep the column
    ephem = ephem.set_index("epoch_utc", drop=False)

    # Sort by time
    ephem = ephem.sort_index()

    df = interpolate_pointing(
        ephem,
        res=input_dict["res"],
        method=input_dict["method"],
        save_df=input_dict["save_df"],
        filename=input_dict["filename"],
        filetype=input_dict["filetype"],
    )

# --------- set inputs to pass to function -------------
start_string = "2024-07-08 18:01:00.00"
start_time = datetime.datetime.strptime(start_string, "%Y-%m-%d %H:%M:%S.%f")
start_time = start_time.replace(tzinfo=datetime.timezone.utc)

stop_string = "2024-07-15 09:01:00.00"
stop_time = datetime.datetime.strptime(stop_string, "%Y-%m-%d %H:%M:%S.%f")
stop_time = stop_time.replace(tzinfo=datetime.timezone.utc)

# Select df rows that are within the time range of interest
# df_lxi = df[(df["epoch_utc"] > start_time) & (df["epoch_utc"] <= stop_time)]

try:
    # Read the exposure map from a pickle file
    exposure = np.load(
        f"../data/exposure_map_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_parallel.npy"
    )
    print("Exposure map loaded from file \n")
except FileNotFoundError:
    print("Exposure map not found, computing now. This may take a while \n")
    # Make array for exposure mask
    x_grid = np.arange(
        input_dict["xrange"][0], input_dict["xrange"][1], input_dict["x_res"]
    )
    y_grid = np.arange(
        input_dict["yrange"][0], input_dict["yrange"][1], input_dict["y_res"]
    )
    x_grid_arr = np.tile(x_grid, (len(y_grid), 1)).transpose()
    y_grid_arr = np.tile(y_grid, (len(x_grid), 1))

    # For each item in the time list, find out its distance from
    # Make an empty exposure map
    exposure = np.zeros((len(x_grid), len(y_grid)))
    zero = np.zeros(exposure.shape)

    # Loop through each pointing step and add the exposure to the map
    # Compute the max number of threads to use
    # The number of cores to use
    num_cores = input_dict["n_cores"]  # mp.cpu_count()

    p = mp.Pool(num_cores)
    input = (
        (
            i,
            x_grid_arr,
            y_grid_arr,
            df.mp_ra[i],
            df.mp_dec[i],
            input_dict["LEXI_FOV"],
            input_dict["step"],
        )
        for i in range(len(df))
    )
    exposure_deltas = p.map(calculate_exposure_delta, input)
    p.close()
    p.join()

    # Sum up the exposure deltas to get the final exposure array
    for exposure_delta in exposure_deltas:
        exposure += exposure_delta

    exposure = exposure / 10.0  # divide by 10 to convert from 100 ms steps to 1s steps

    # Save the exposure map to a pickle file
    np.save(
        f"../data/exposure_map_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_test.npy",
        exposure,
    )

max_exposure = np.nanmax(exposure)
min_exposure = np.nanmin(exposure)

# Set the min and max values for the colorbar
v_min = 0.9 * np.nanmin(exposure)
v_max = 1.1 * np.nanmax(exposure)

fig = plt.figure()
ax = plt.axes()

plt.xlabel("RA [$^\\circ$]")
plt.ylabel("DEC [$^\\circ$]")
plt.title(f"LEXI: {start_string} - {stop_string}")

im = ax.imshow(
    np.transpose(exposure),
    cmap="Blues",
    extent=[
        input_dict["xrange"][0],
        input_dict["xrange"][1],
        input_dict["yrange"][0],
        input_dict["yrange"][1],
    ],
    origin="lower",
    aspect="equal",
    norm=cm.colors.Normalize(vmin=v_min, vmax=v_max),
)

# Add n_steps and step as text to the plot
plt.text(
    0.05,
    0.95,
    f"n_steps = {len(df)} \n step = {input_dict['step']} s \n Exposure = [{min_exposure:.2f}, {max_exposure:.2f}]",
    horizontalalignment="left",
    verticalalignment="top",
    transform=ax.transAxes,
    fontsize=12,
)

cax = fig.add_axes(
    [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height]
)
plt.colorbar(im, cax=cax, label="Time in each pixel [s]")

plt.savefig(
    f"../figures/exposure_map_test_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_parallel_v2.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
# plt.show()

code_end_time = time.time()
print(f"Code took {np.round(code_end_time - code_start_time, 3)} seconds to run")
