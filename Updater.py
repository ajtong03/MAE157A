import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.animation as animation  
from mpl_toolkits.mplot3d import Axes3D  
from math import sin, cos
from quaternionfunc import quat_to_euler

class Updater:
    def __init__(self):
        self.history = []
        self.axis_arrows = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.trail, = self.ax.plot([], [], [], 'b-', lw=2)
        self.arm1, = self.ax.plot([], [], [], 'k-', lw=2)
        self.arm2, = self.ax.plot([], [], [], 'k-', lw=2)
    def initializePlot(self):
        self.ax.set_xlim(-2, 2); self.ax.set_ylim(-2, 2); self.ax.set_zlim(0, 4)
        # gate at z=1 rotated 90° about Y
        gate = np.array([[-.5,0,-.25],[.5,0,-.25],[.5,0,.25],[-.5,0,.25],[-.5,0,-.25]])
        ty = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        gate_pts = gate @ ty.T + np.array([0,0,1])
        self.ax.plot(gate_pts[:,0], gate_pts[:,1], gate_pts[:,2], 'g-', lw=2)

    def updatePlot(self, state, dyn):
        self.history.append(state[0:3].copy())
        pts = np.array(self.history)
        self.trail.set_data(pts[:,0], pts[:,1]); self.trail.set_3d_properties(pts[:,2])
        # draw arms based on new orientation
        phi, th, ps = quat_to_euler(state[6:10])
        offs = [(dyn.l,0,0),(-dyn.l,0,0),(0,dyn.l,0),(0,-dyn.l,0)]
        world = []
        for dx,dy,dz in offs:
            Xw = cos(ps)*cos(th)*dx + (cos(ps)*sin(phi)*sin(th)-cos(phi)*sin(ps))*dy \
                + (cos(phi)*cos(ps)*sin(th)+sin(phi)*sin(ps))*dz + state[0]
            Yw = sin(ps)*cos(th)*dx + (sin(ps)*sin(phi)*sin(th)+cos(phi)*cos(ps))*dy \
                + (cos(phi)*sin(ps)*sin(th)-cos(ps)*sin(phi))*dz + state[1]
            Zw = -sin(th)*dx + cos(th)*sin(phi)*dy + cos(phi)*cos(th)*dz + state[2]
            world.append((Xw,Yw,Zw))
        a,b,c,d = world
        self.arm1.set_data([a[0],c[0]],[a[1],c[1]]); self.arm1.set_3d_properties([a[2],c[2]])
        self.arm2.set_data([b[0],d[0]],[b[1],d[1]]); self.arm2.set_3d_properties([b[2],d[2]])
        # update orientation arrows
        for art in self.axis_arrows:
            art.remove()
        self.axis_arrows = []
        Rmat = dyn.quat_to_rot(state[6:10])
        pos = state[0:3]
        self.axis_arrows.append(self.ax.quiver(pos[0],pos[1],pos[2],Rmat[0,0],Rmat[1,0],Rmat[2,0],length=0.3))
        self.axis_arrows.append(self.ax.quiver(pos[0],pos[1],pos[2],Rmat[0,1],Rmat[1,1],Rmat[2,1],length=0.3))
        self.axis_arrows.append(self.ax.quiver(pos[0],pos[1],pos[2],Rmat[0,2],Rmat[1,2],Rmat[2,2],length=0.3))
        return self.trail, self.arm1, self.arm2