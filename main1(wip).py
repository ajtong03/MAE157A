
# main.py
import numpy as np
import os
import datetime
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dynamics

# --- Trajectory definitions inline ---
gate = None
T_total = 8.0  # total flight time (s)

def load_gate(path=None):
    global gate
    gate = {
        'position': np.array([5.0, 0.0, 2.0]),
        'yaw': np.deg2rad(90.0),
        'pitch': 0.0  # no pitch, only yaw for a true rectangle
    }
    return gate


def setup_trajectory(g):
    global gate
    gate = g


def total_time():
    return T_total


def desired_state(t):
    p0 = np.zeros(3)
    pg = gate['position']
    tau = T_total / 2
    if t <= tau:
        s = t / tau
        pos = p0 + s * (pg - p0)
        vel = (pg - p0) / tau
        acc = np.zeros(3)
    else:
        s = (t - tau) / tau
        pos = pg + s * (pg - p0)
        vel = (pg - p0) / tau
        acc = np.zeros(3)
    yaw = gate['yaw']
    yaw_rate = 0.0
    return pos, vel, acc, yaw, yaw_rate

# --- Attitude helper functions ---
def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        S = 2 * np.sqrt(t + 1.0)
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            S = 2 * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / S; x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S; z = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S = 2 * np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S; y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = 2 * np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S; z = 0.25 * S
    return np.array([w, x, y, z])

def attitude_from_acc(a_des, yaw):
    b3 = a_des / np.linalg.norm(a_des)
    c = np.cos(yaw); s = np.sin(yaw)
    b1d = np.array([c, s, 0.0])
    b2 = np.cross(b3, b1d); b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    R_des = np.column_stack((b1, b2, b3))
    return rot_to_quat(R_des)

# --- Controllers inline ---
def default_takeoff_controller(t, state):
    z_des = 10.0; mass = 1.0; g = 9.8
    err = z_des - state[2]; err_dot = -state[5]
    F = mass * (g + 2.0 * err + 1.0 * err_dot)
    F = max(0.0, F)
    return np.array([F/4.0] * 4)

# --- Simulation ---
def run_simulation(use_trajectory):
    t = 0.0
    rate = 500
    dt = 1.0 / rate
    tf = total_time()
    g = 9.8
    m = 0.399

    state = np.zeros(13)
    state[6] = 1.0  # quaternion w
    state[2] = 0.1  # start 0.1 m above ground
    f = np.zeros(4)

    dyn = dynamics.dynamics(np.array([g]), dt)
    dyn.print_specs()

    history = []
    while t < tf:
        # Select motor forces
        if use_trajectory:
            # takeoff phase
            if state[2] < 0.5:
                f = default_takeoff_controller(t, state)
            else:
                # trajectory phase
                pd, vd, ad, yawd, _ = desired_state(t)
                e = pd - state[0:3]
                ed = vd - state[3:6]
                a_des = ad + np.diag([1.5,1.5,2.0]) @ e + np.diag([0.8,0.8,1.0]) @ ed + np.array([0,0,g])
                T = max(m * np.linalg.norm(a_des), m * g)
                l = dyn.l
                My = a_des[0] * m * l
                f_base = T / 4.0
                f1 = np.clip(f_base - My/(2*l), 0.0, None)
                f2 = np.clip(f_base, 0.0, None)
                f3 = np.clip(f_base + My/(2*l), 0.0, None)
                f4 = np.clip(f_base, 0.0, None)
                f = np.array([f1, f2, f3, f4])
        else:
            f = default_takeoff_controller(t, state)

        # Propagate one step
        state = dyn.propagate(state, f, dt)
        history.append((t, state.copy(), f.copy()))

        # Advance time
        t += dt

    # Save trajectory
    times = np.array([h[0] for h in history])
    posns = np.array([h[1][0:3] for h in history])
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = f"traj_sim_{now}.csv"
    save_dir = r"C:/Users/TechP/Documents"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    np.savetxt(path,
               np.hstack((times.reshape(-1,1), posns)),
               delimiter=',',
               header='t,x,y,z',
               comments='')
    print(f"Saved to {path}")

    # Plot trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(posns[:,0], posns[:,1], posns[:,2], linewidth=3, label='Drone Path')
    # Draw rectangular gate
    corners = np.array([
        [-0.25,0,-0.5],[0.25,0,-0.5],[0.25,0,0.5],[-0.25,0,0.5],[-0.25,0,-0.5]
    ])
    yaw = gate['yaw']
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1],
    ])
    world_c = (Rz @ corners.T).T + gate['position']
    ax.plot(world_c[:,0], world_c[:,1], world_c[:,2], color='g', linewidth=4, label='Gate')

    ax.set(xlabel='X', ylabel='Y', zlabel='Z', title='Drone Trajectory Through Gate')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drone gate-tracking sim.')
    parser.add_argument('--ctrl', choices=['trajectory','default'], default='trajectory')
    args = parser.parse_args()

    g = load_gate(None)
    setup_trajectory(g)
    run_simulation(args.ctrl == 'trajectory')

