import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.animation as animation  
from mpl_toolkits.mplot3d import Axes3D  
from math import sin, cos
from quaternionfunc import quat_to_euler, quat_to_rot
from trajectory import traj

class Updater:
    def __init__(self):
        self.history = []
        self.axis_arrows = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.trail, = self.ax.plot([], [], [], 'm-', lw=2)
        self.arm1, = self.ax.plot([], [], [], 'k-', lw=2)
        self.arm2, = self.ax.plot([], [], [], 'k-', lw=2)
    
    def initializePlot(self):
        self.ax.set_xlim(-2, 2)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 4]) 
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('3D Drone Trajectory through Gate')
        self.ax.grid(True)
        self.ax.plot([0], [-1], [1.5], 'ro', markersize=5, label='Gate Origin')  # gate at (0,0,1)

        gate = np.array([
            [0, -0.25, -0.1905],
            [0,  0.25, -0.1905],
            [0,  0.25,  0.1905],
            [0, -0.25,  0.1905],
            [0, -0.25, -0.1905]  
        ])

        # 45-degree rotation about Y-axis for gate
        theta = np.radians(45)
        ty = np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta) ],
            [0, -np.sin(theta), np.cos(theta)]
        ])

        # Rotate and translate gate to origin at (0,0,1)
        initial_normal = np.array([0, 0, 1])  # Normal 
        gate_pts = gate @ ty.T + np.array([0, -1, 1.5])
        #gate_normal = np.array([0, 0.5, 1.75]) @ ty.T
        gate_normal = ty @ initial_normal  # Rotate the normal, not the origin
        gate_normal = gate_normal / np.linalg.norm(gate_normal)
        
        self.ax.plot(gate_pts[:, 0], gate_pts[:, 1], gate_pts[:, 2], color = 'black', lw=2)
        self.ax.quiver(
            0, -1, 1.5,                    
            gate_normal[0], gate_normal[1], gate_normal[2],  # Components
            length=0.5, color='purple', linewidth=2, label='Gate Normal'
        )
        self.ax.plot(gate_pts[:, 0], gate_pts[:, 1], gate_pts[:, 2], color = 'black', lw=2)
        self.ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], '--', color = 'r', lw= 0.75, label = 'trajectory path')
        self.ax.legend(facecolor = 'lightgrey')

        

    def updateTrail(self, state):
        self.history.append(state[0:3].copy())

        pts = np.array(self.history)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        
        # update trail line
        self.trail.set_data(pts[:, 0], pts[:, 1])
        self.trail.set_3d_properties(pts[:, 2])
        
        return [self.trail]
    

    def updateDrone(self, state, dyn):
        # draw arms based on new orientation
        offs = [np.array([dyn.l, 0, 0]),
                np.array([-dyn.l, 0, 0]),
                np.array([0, dyn.l, 0]),
                np.array([0, -dyn.l, 0])]

        R = quat_to_rot(state[6:10])
        pos = state[0:3]
        world = [R @ offset + pos for offset in offs]

        a,b,c,d = world
        self.arm1.set_data([a[0],c[0]],[a[1],c[1]])
        self.arm1.set_3d_properties([a[2],c[2]])
        self.arm2.set_data([b[0],d[0]],[b[1],d[1]])
        self.arm2.set_3d_properties([b[2],d[2]])

        # update orientation arrows
        for art in self.axis_arrows:
            art.remove()
        self.axis_arrows = []

        pos = state[0:3]
        self.axis_arrows.append(self.ax.quiver(pos[0],pos[1],pos[2],R[0,0], R[1,0], R[2,0], color = 'r', length=0.3))
        self.axis_arrows.append(self.ax.quiver(pos[0],pos[1],pos[2],R[0,1],R[1,1],R[2,1], color = 'g', length=0.3))
        self.axis_arrows.append(self.ax.quiver(pos[0],pos[1],pos[2],R[0,2],R[1,2],R[2,2], color = 'b', length=0.3))
        
        return [self.arm1, self.arm2] + self.axis_arrows