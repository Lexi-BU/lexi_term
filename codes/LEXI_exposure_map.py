# 2023 Jun 29 - BMW
# Known issue!  the code only appears to work when there is evenly sized x and y axis...

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

# ------- Read in sample made up ephemeris file and manipulate ------
path = '/Users/bwalsh/Documents/Research/LEXI/Software/exposuremap/'
ephem = pd.read_csv(path+'SAMPLE_LEXI_pointing_ephem.csv',sep=',')

# convert time and pointing to np array
time = pd.to_datetime((ephem['[Epoch (UTC)]']).to_numpy())
mp_RA_deg = ephem['[Magnetopause Track Ra (deg)]'].to_numpy()
mp_DEC_deg = ephem['[Magnetopause Track Dec (deg)]'].to_numpy()

# interpolate the even time steps to get 100 ms resolution
n = time.size
steps = n*60*10 #number of 100 ms steps in 1 minute
#time_highres = np.linspace(min(time).value, max(time).value, steps)
time_highres = np.linspace(min(time).value, max(time).value, steps)
time_highres = pd.to_datetime(time_highres)

mp_RA_deg_highres = np.interp(time_highres,time,mp_RA_deg)
mp_DEC_deg_highres = np.interp(time_highres,time,mp_DEC_deg)
#---------- Done manipulating the pointing data ----------------


# --------- set inputs to pass to function -------------
start_string = '2024-07-08T18:01:00.000000000'
stop_string = '2024-07-09T09:01:00.000000000'

# right now this works with square even pixels
x_res = 1.0 # res in degrees
y_res = 1.0 # res in degrees
LEXI_FOV = 9.1 # deg
roll = 0.0 # deg roll angle.  Here 0 deg will correspond to line up perfectly with RA/DEC
xrange = [315.0,355.0] # desired input for plotting ranges
yrange = [-24.0,6.0] # desired input for plotting ranges
# xoffset = 0.0 # deg angle from Az of mounting plate value to RA
# yoffset = 9.1/2.0 # deg angle from El of mounting plate value to DEC
step  = 0.1


# --------------- Run things inside function ---------------
# -------- input parts to run mask ----------------
tstart =pd.to_datetime(start_string)
tstop = pd.to_datetime(stop_string)

def vignette(d):
    f = 1 #Vignetting factor distance d (in degrees) from boresight (use numpy operations for speed)
    return f

# Trim arrays to the desired time range
time_mask = ((time_highres > tstart) & (time_highres <= tstop)) #Times inside the time range
mp_RA_deg_highres = mp_RA_deg_highres[time_mask]
mp_DEC_deg_highres = mp_DEC_deg_highres[time_mask]
time_highres = time_highres[time_mask]

# Make array for exposure mask
x_grid = np.arange(xrange[0], xrange[1], x_res)
y_grid = np.arange(yrange[0], yrange[1], y_res)
x_grid_arr = np.tile(x_grid, (len(y_grid),1)).transpose()
y_grid_arr = np.tile(y_grid, (len(x_grid),1))

# Make an empty exposure map
exposure = np.zeros((len(x_grid),len(y_grid)))
zero = np.zeros(exposure.shape)

print(min(mp_RA_deg_highres))
print(max(mp_RA_deg_highres))

for i in range(len(time_highres)):
    r = np.sqrt((x_grid_arr - mp_RA_deg_highres[i])**2+(y_grid_arr - mp_DEC_deg_highres[i])**2) #Get distance in degrees to the pointing step
    exposure_delt = np.where((r < LEXI_FOV*0.5),vignette(r)*step, zero) #Make an exposure delta for this span
    exposure += exposure_delt #Add the delta to the full map
# #return exposure

exposure = exposure/10. # divide by 10 to convert from 100 ms steps to 1s steps

fig=plt.figure()
ax = plt.axes()

plt.xlabel('RA [Deg]')
plt.ylabel('DEC [Deg]')
plt.title('LEXI, 1.0 deg pix, 2024-07-08T18:01:00 - 2024-07-09T09:01:00')

im = ax.imshow(exposure,cmap='Blues',extent=[xrange[0], xrange[1],yrange[0], yrange[1]], aspect = "auto")

cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax,label='Time in bin [s]') # Similar to fig.colorbar(im, cax = cax)

plt.show()