import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

# gc.dat에서 2,3,4열 읽기
gc_data = np.loadtxt('gc.dat', usecols=(1,2,3))
x, y, z = gc_data[:,0], gc_data[:,1], gc_data[:,2]

# newton.dat에서 2,3,4열 읽기
newton_data = np.loadtxt('newton.dat', usecols=(1,2,3))
x2, y2, z2 = newton_data[:,0], newton_data[:,1], newton_data[:,2]

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='gc.dat', marker='o')
ax.plot(x2, y2, z2, label='newton.dat', marker='^')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()