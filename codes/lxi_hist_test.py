import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

hist_dat = global_variables.data_lin["counts"]
x_edges = global_variables.data_lin["xedges"]
y_edges = global_variables.data_lin["yedges"]

plt.close("all")
plt.figure(figsize=(5, 5))
im = plt.imshow(hist_dat.T, interpolation='nearest', origin='lower', extent=[x_edges[0], x_edges[-1],
                y_edges[0], y_edges[-1]], cmap="Reds", norm=mpl.colors.LogNorm(
                vmin=1, vmax=np.nanmax(hist_dat)))
plt.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04, label='Counts')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('../figures/lxi_hist_test.png', dpi=300, bbox_inches='tight', pad_inches=0.1)