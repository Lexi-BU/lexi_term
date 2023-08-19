import bokeh as bk
import datetime
import importlib
import time

import LEXI_exposure_map_fnc as lemf

import numpy as np

importlib.reload(lemf)

# Close any open plots
bk.io.curdoc().clear()

code_start_time = time.time()

input_dict = {
    "t_res": "100L",  # Time resolution to interpolate to. Default is 100 ms
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
}

# Try to read in the high res pointing file, if it doesn't exist then make it
df = lemf.read_create_df(input_dict)

# --------- set inputs to pass to function -------------
start_string = "2024-07-08 18:01:00.00"
start_time = datetime.datetime.strptime(start_string, "%Y-%m-%d %H:%M:%S.%f")
start_time = start_time.replace(tzinfo=datetime.timezone.utc)

stop_string = "2024-07-15 09:01:00.00"
stop_time = datetime.datetime.strptime(stop_string, "%Y-%m-%d %H:%M:%S.%f")
stop_time = stop_time.replace(tzinfo=datetime.timezone.utc)

# Select df rows that are within the time range of interest
# df_lxi = df[(df["epoch_utc"] > start_time) & (df["epoch_utc"] <= stop_time)]

exposure = lemf.exposure_map(input_dict=input_dict, df=df, save_map=True)

# Save the exposure map to a png file
_ = lemf.matplotlib_figure(
    df=df,
    input_dict=input_dict,
    exposure=exposure,
    start_string=start_string,
    stop_string=stop_string,
    display=False,
    figure_format="pdf",
)

# Save the exposure map to an interactive html file

_ = lemf.bokeh_figure(
    input_dict=input_dict,
    exposure=exposure,
    start_string=start_string,
    stop_string=stop_string,
    display=True,
)

code_end_time = time.time()
print(f"\nCode took {np.round(code_end_time - code_start_time, 3)} seconds to run")
