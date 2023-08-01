# 2023 Jun 29 - BMW
# Known issue!  the code only appears to work when there is evenly sized x and y axis...

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

import time

code_start_time = time.time()
# ------- Read in sample made up ephemeris file and manipulate ------
# path = '/Users/bwalsh/Documents/Research/LEXI/Software/exposuremap/'
ephem = pd.read_csv("SAMPLE_LEXI_pointing_ephem_edited.csv", sep=",")

# Set 'epoch_utc' column to datetime object, and set time to UTC
ephem["epoch_utc"] = pd.to_datetime(ephem["epoch_utc"], utc=True)

# Set index to 'epoch_utc' column, also keep the column
ephem = ephem.set_index("epoch_utc", drop=False)

# Sort by time
ephem = ephem.sort_index()

# Keys in the ephemeris file
key_list = [
    "epoch_utc",
    "epoch_mjd",
    "earth_ra",
    "earth_dec",
    "sun_ra",
    "sun_dec",
    "sco_ra",
    "sco_dec",
    "mp_az",
    "mp_el",
    "mp_ra",
    "mp_dec",
]

# convert time and pointing to np array
time_utc = ephem["epoch_utc"]
mp_RA_deg = ephem["mp_ra"]
mp_DEC_deg = ephem["mp_dec"]

# interpolate the even time steps to get 100 ms resolution
n = time_utc.size
n_steps = int(n / 10)  # number of 100 ms steps in 1 minute
# Find n_steps datetimes between min and max time
time_hres_list = np.linspace(
    min(time_utc).timestamp(), max(time_utc).timestamp(), n_steps
)
datetime_hres_list = np.array(
    [datetime.datetime.fromtimestamp(x) for x in time_hres_list]
)
time_highres = pd.to_datetime(datetime_hres_list, utc=True)

# Interpolate the pointing data to the highres time steps
mp_RA_deg_highres = np.interp(time_highres, time_utc, mp_RA_deg)
mp_DEC_deg_highres = np.interp(time_highres, time_utc, mp_DEC_deg)


# Make a dataframe of the highres time and pointing data
df = pd.DataFrame(
    {
        "time_utc": time_highres,
        "mp_RA_deg": mp_RA_deg_highres,
        "mp_DEC_deg": mp_DEC_deg_highres,
    }
)

# ---------- Done manipulating the pointing data ----------------


# --------- set inputs to pass to function -------------
start_string = "2024-07-08 18:01:00.00"
start_time = datetime.datetime.strptime(start_string, "%Y-%m-%d %H:%M:%S.%f")
start_time = start_time.replace(tzinfo=datetime.timezone.utc)

stop_string = "2024-07-15 09:01:00.00"
stop_time = datetime.datetime.strptime(stop_string, "%Y-%m-%d %H:%M:%S.%f")
stop_time = stop_time.replace(tzinfo=datetime.timezone.utc)

# right now this works with square even pixels
x_res = 0.1  # res in degrees
y_res = 0.1  # res in degrees
LEXI_FOV = 9.1  # deg
roll = (
    0.0  # deg roll angle.  Here 0 deg will correspond to line up perfectly with RA/DEC
)
xrange = [325.0, 365.0]  # desired input for plotting ranges
yrange = [-21.0, 6.0]  # desired input for plotting ranges
# xoffset = 0.0 # deg angle from Az of mounting plate value to RA
# yoffset = 9.1/2.0 # deg angle from El of mounting plate value to DEC
step = 0.01


# --------------- Run things inside function ---------------
# -------- input parts to run mask ----------------


def vignette(d):
    f = 1  # Vignetting factor distance d (in degrees) from boresight (use numpy operations for speed)
    return f


# Select df rows that are within the time range of interest
df = df[(df["time_utc"] > start_time) & (df["time_utc"] <= stop_time)]

# Get the histogram of the RA and DEC values
xedges = np.arange(xrange[0], xrange[1], x_res)
yedges = np.arange(yrange[0], yrange[1], y_res)
H, xedges, yedges = np.histogram2d(
    df["mp_RA_deg"], df["mp_DEC_deg"], bins=(xedges, yedges)
)

# Make array for exposure mask
x_grid = np.arange(xrange[0], xrange[1], x_res)
y_grid = np.arange(yrange[0], yrange[1], y_res)
x_grid_arr = np.tile(x_grid, (len(y_grid), 1)).transpose()
y_grid_arr = np.tile(y_grid, (len(x_grid), 1))

# For each item in the time list, find out its distance from
# Make an empty exposure map
exposure = np.zeros((len(x_grid), len(y_grid)))
zero = np.zeros(exposure.shape)

# Count the number of

for i in range(len(time_highres)):
    # If i is divisible bt 10, then print the progress
    # if i % 1000 == 0:
    #     print("Progress: ", i, " out of ", len(time_highres))
    # for j in range(len(x_grid)):
    #     for k in range(len(y_grid)):
    #         r = np.sqrt(
    #             (x_grid[j] - mp_RA_deg_highres[i]) ** 2
    #             + (y_grid[k] - mp_DEC_deg_highres[i]) ** 2
    #         )
    #         if r < LEXI_FOV * 0.5:
    #             exposure[j, k] += vignette(r) * step

    r = np.sqrt(
        (x_grid_arr - mp_RA_deg_highres[i]) ** 2
        + (y_grid_arr - mp_DEC_deg_highres[i]) ** 2
    )  # Get distance in degrees to the pointing step
    exposure_delt = np.where(
        (r < LEXI_FOV * 0.5), vignette(r) * step, 0
    )  # Make an exposure delta for this span
    exposure += exposure_delt  # Add the delta to the full map
# #return exposure

exposure = exposure / 10.0  # divide by 10 to convert from 100 ms steps to 1s steps

print(np.nanmax(exposure))
fig = plt.figure()
ax = plt.axes()

plt.xlabel("RA [Deg]")
plt.ylabel("DEC [Deg]")
plt.title("LEXI, 1.0 deg pix, 2024-07-08T18:01:00 - 2024-07-09T09:01:00")

im = ax.imshow(
    np.transpose(exposure),
    cmap="Blues",
    extent=[xrange[0], xrange[1], yrange[0], yrange[1]],
    origin="lower",
    aspect="equal",
)

cax = fig.add_axes(
    [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height]
)
plt.colorbar(
    im, cax=cax, label="Time in bin [s]"
)  # Similar to fig.colorbar(im, cax = cax)

plt.savefig("../figures/exposure_map_v4.png", dpi=300)
# plt.show()

code_end_time = time.time()
print(f"Code took {np.round(code_end_time - code_start_time, 3)} seconds to run")
