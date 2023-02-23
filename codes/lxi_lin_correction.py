import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([0.4325, 0.449, 0.4675, 0.4875, 0.505]) - 0.5
y = np.array([0.5175, 0.516, 0.5135, 0.511, 0.5095]) - 0.5

# x = np.array([0.444, 0.4515, 0.46, 0.47, 0.4795, 0.498, 0.5175])
# y = np.array([0.50875, 0.5065, 0.505, 0.5045, 0.503, 0.501, 0.5025])

# Find the slope and intercept of the line
m, b = np.polyfit(x, y, 1)

# Get the angle of the line
angle = np.arctan(m)

# Get the roration matrix
R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
# Get the inverse rotation matrix
R_inv = np.linalg.inv(R)

# Rotate the data
# x_rot = R_inv[0, 0] * x + R_inv[0, 1] * y - R_inv[0, 0] * b
y_rot = R_inv[1, 0] * x + R_inv[1, 1] * y - R_inv[1, 1] * b

x1 = np.array([0.463, 0.465, 0.4675, 0.47, 0.4705, 0.475]) - 0.5
y1 = np.array([0.552, 0.532, 0.5135, 0.505, 0.495, 0.4775]) - 0.5

# Find the slope and intercept of the line
m1, b1 = np.polyfit(x1, y1, 1)

# Get the angle of the line
angle1 = np.arctan(m1)

# Get the roration matrix
R1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])

# Get the inverse rotation matrix
R_inv1 = np.linalg.inv(R1)

R_inv_f = np.matmul(R_inv, R_inv1)

# Rotate the data
x_rot1 = R_inv_f[1, 0] * x1 + R_inv_f[1, 1] * y1 - R_inv_f[0, 0] * b1
# y_rot1 = R_inv1[1, 0] * x1 + R_inv1[1, 1] * y1 - R_inv1[1, 1] * b1

x_rot = R_inv_f[1, 0] * x + R_inv_f[1, 1] * y - R_inv_f[0, 0] * b1
y_rot1 = R_inv_f[1, 0] * x1 + R_inv_f[1, 1] * y1 - R_inv_f[1, 1] * b

# Plot the data
plt.close("all")
plt.figure()
plt.plot(x, y, 'o', label='data')
plt.plot(x_rot, y_rot, 'o', label='corrected data')
plt.legend()
plt.savefig('../figures/test.png')

# Plot the data
plt.close("all")
plt.figure()
plt.plot(x1, y1, 'o', label='data')
plt.plot(x_rot1, y_rot1, 'o', label='corrected data')
plt.legend()
plt.savefig('../figures/test1.png')

# np.array([[0.98678, 0.16204], [0.11392, 0.99349]])

# 
print([[R_inv1[1,0], R_inv1[1,1]], [R_inv[1,0], R_inv[1,1]]])