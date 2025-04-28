import numpy as np
import matplotlib.pyplot as plt
import os

# Automatically find the latest simulation CSV in Documents:
folder = r"C:\Users\Sanat\Documents"
csvs = [f for f in os.listdir(folder) if f.startswith('traj_sim_') and f.endswith('.csv')]
if not csvs:
    raise FileNotFoundError("No traj_sim_*.csv files found in Documents")
csvs.sort()
file_path = os.path.join(folder, csvs[-1])

# Load data: columns t,x,y,z
data = np.loadtxt(file_path, delimiter=',', skiprows=1)

time = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

# Plot altitude over time
plt.figure()
plt.plot(time, z, label='Altitude')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Drone Altitude vs Time')
plt.grid(True)
plt.legend()
plt.show()
