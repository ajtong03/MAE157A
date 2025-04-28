import numpy as np
import quaternionfunc

def quat_angle_err(q_act, q_d):
    """Compute the scalar angle error between two quaternions."""
    q_err = quaternionfunc.error(q_act, q_d)
    q_err /= np.linalg.norm(q_err)
    return 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))


def overshoot_settime(q_act, q_d, time=None, thresh=0.05):
    """
    Compute overshoot and settling time of a quaternion sequence.

    Args:
        q_act: list or array of unit quaternions (N×4)
        q_d: desired unit quaternion (4,)
        time: None, scalar total time, or array of timestamps length N
        thresh: settling threshold in radians

    Returns:
        overshoot (float), settling_time (float)
    """
    angle_err = np.array([quat_angle_err(q, q_d) for q in q_act])
    overshoot = np.max(angle_err)

    n = len(angle_err)
    if isinstance(time, (int, float)):
        t_arr = np.linspace(0, time, n)
    elif hasattr(time, '__len__') and len(time) == n:
        t_arr = np.array(time)
    else:
        t_arr = np.arange(n)

    set_time = t_arr[-1]
    for i in range(n):
        if np.all(angle_err[i:] < thresh):
            set_time = t_arr[i]
            break

    return overshoot, set_time


def naiveComputeGains(q_act, q_d):
    """Placeholder for a naive gain search."""
    raise NotImplementedError("naiveComputeGains is not implemented yet.")


def computeGains(q_act, q_d):
    """
    Tune diagonal PD gains for quaternion convergence using random search.

    Args:
        q_act: list or array of past quaternions (N×4)
        q_d: desired quaternion (4,)

    Returns:
        Kp_opt: optimal proportional gain matrix (3×3)
        Kd_opt: optimal derivative gain matrix (3×3)
        overshoot: overshoot of the best trial
        settling_time: settling time of the best trial
    """
    # initial search ranges
    Kp_range = (1.0, 20.0)
    Kd_range = (1.0, 20.0)

    # default gains (identity)
    Kp_opt = np.eye(3)
    Kd_opt = np.eye(3)

    # baseline performance
    over_opt, set_opt = overshoot_settime(q_act, q_d)
    perf_opt = over_opt + set_opt

    for _ in range(1000):
        Kp = np.diag(np.random.uniform(Kp_range[0], Kp_range[1], size=3))
        Kd = np.diag(np.random.uniform(Kd_range[0], Kd_range[1], size=3))

        overshoot, settle = overshoot_settime(q_act, q_d)
        perf = overshoot + settle

        if perf < perf_opt:
            perf_opt = perf
            Kp_opt = Kp
            Kd_opt = Kd
            over_opt = overshoot
            set_opt = settle
            # narrow search around the best
            mu = Kp_opt[0, 0]
            Kp_range = (mu - 0.5, mu + 0.5)
            mu = Kd_opt[0, 0]
            Kd_range = (mu - 0.5, mu + 0.5)

    return Kp_opt, Kd_opt, over_opt, set_opt


def computeLambda(lambda_range, lambda_opt):
    """Placeholder for lambda tuning (not used)."""
    return np.eye(3)


def computeTorque(Kp, Kd, lambda_opt, q_act, q_d, w, w_d, Re):
    """
    Compute a PD control torque from quaternion error and body rate error.

    Args:
        Kp: proportional gain matrix (3×3)
        Kd: derivative gain matrix (3×3)
        lambda_opt: unused placeholder
        q_act: actual quaternion (4,)
        q_d: desired quaternion (4,)
        w: current angular rate (3,)
        w_d: desired angular rate (3,)
        Re: rotation error mapping (3×3), typically quat_to_rot(q_err)

    Returns:
        torque: control torque vector (3,)
    """
    # quaternion error
    qe = quaternionfunc.error(q_act, q_d)
    qe /= np.linalg.norm(qe)
    qe_mag = qe[0]
    qe_vec = qe[1:]
    # rate error
    w_e = w - Re.dot(w_d)
    # PD torque
    torque = -qe_mag * (Kp.dot(qe_vec)) - Kd.dot(w_e)
    return torque
