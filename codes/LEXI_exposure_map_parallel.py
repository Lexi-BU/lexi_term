import datetime
import importlib
import multiprocessing as mp
import resource
import time

import LEXI_exposure_map_parallel_fnc as lepf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from text_color_fnc import text_color as tc

importlib.reload(lepf)

tc = tc()

# Set latex use to true
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


# Set the maximum amount of memory to 90% of available memory
mem_limit = int(resource.getrlimit(resource.RLIMIT_AS)[1] * 1)
resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

# Set the start time
code_start_time = time.time()

input_dict = {
    "t_res": "100L",  # Time resolution to interpolate to.  Default is 100 ms
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
    "n_cores": 1,  # number of cores to use
}

# Try to read in the high res pointing file, if it doesn't exist then make it
try:
    df = pd.read_pickle(
        f"../data/LEXI_pointing_ephem_highres_tres_{input_dict['t_res']}.pkl"
    )
    print(tc.green_text("\n High res pointing file loaded from file \n"))
except FileNotFoundError:
    print(
        tc.red_text(
            "High res pointing file not found, computing now. This may take a while \n"
        )
    )
    ephem = pd.read_csv("SAMPLE_LEXI_pointing_ephem_edited.csv", sep=",")

    # Set 'epoch_utc' column to datetime object, and set time to UTC
    ephem["epoch_utc"] = pd.to_datetime(ephem["epoch_utc"], utc=True)

    # Set index to 'epoch_utc' column, also keep the column
    ephem = ephem.set_index("epoch_utc", drop=False)

    # Sort by time
    ephem = ephem.sort_index()

    df = lepf.interpolate_pointing(
        ephem,
        t_res=input_dict["t_res"],
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
        f"../data/exposure_map_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_tres_{input_dict['t_res']}_parallel_v_v.npy"
    )
    print(tc.green_text("Exposure map loaded from file \n"))
except FileNotFoundError:
    print(
        tc.red_text("Exposure map not found, computing now. This may take a while \n")
    )
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

    p = mp.Pool()
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
    exposure_deltas = p.map(lepf.calculate_exposure_delta, input)
    p.close()
    p.join()

    # Sum up the exposure deltas to get the final exposure array
    for exposure_delta in exposure_deltas:
        exposure += exposure_delta

    exposure = exposure / 10.0  # divide by 10 to convert from 100 ms steps to 1s steps

    # Save the exposure map to a pickle file
    save_file = f"../data/exposure_map_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_tres_{input_dict['t_res']}_parallel_v_v.npy"
    np.save(
        save_file,
        exposure,
    )

    print(f"Exposure map saved to file : {tc.green_text(save_file)} \n")

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
    f"../figures/exposure_map_test_xres_{input_dict['x_res']}_yres_{input_dict['y_res']}_tres_{input_dict['t_res']}_parallel_v_v.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
# plt.show()

code_end_time = time.time()
print(
    f"Code took {tc.red_text(np.round(code_end_time - code_start_time, 3))} seconds to run"
)
