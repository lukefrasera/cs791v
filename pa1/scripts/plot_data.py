from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv, sys

x = []
y = []
z = []
with open(sys.argv[1],'rb') as csvfile:
  datareader = csv.reader(csvfile, delimiter=',')
  for row in datareader:
    x.append(int(row[0]))
    y.append(int(row[1]))
    z.append(sum(map(float,row[2:]))/5.0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
ax.scatter(x, y, z)

plt.show()

