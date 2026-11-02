"""Energy-optimal pick of a box with a 4-DOF manipulator in MuJoCo.

Pipeline:  IK (way-points)  ->  energy-optimal trajectory optimisation  ->  computed-torque tracking.

Run:
    python main.py                # with interactive viewer
    python main.py --headless     # no viewer, just results + plots
    python main.py --video        # also save results/pick.mp4 + pick.gif (offscreen rendering)
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from kinematics import solve_ik
from trajopt import EnergyOptimalPlanner, Trajectory

XML = os.path.join(os.path.dirname(__file__), "model", "arm.xml")
ARM_JOINTS = ["j1_yaw", "j2_shoulder", "j3_elbow", "j4_wrist"]
ARM_ACTS = ["m1_yaw", "m2_shoulder", "m3_elbow", "m4_wrist"]
FINGER_ACTS = ["finger_left", "finger_right"]
Q_HOME = np.array([0.0, -0.3, 1.2, 1.0])
GRIP_OPEN, GRIP_CLOSED = 0.0, 0.045
KP, KD = 400.0, 40.0                                   # tracking gains (wn = 20 rad/s, zeta = 1)


def full_inertia(model, data, out):
    """Dense joint-space inertia matrix (handles the mj_fullM signature change in newer MuJoCo)."""
    try:
        mujoco.mj_fullM(model, data, out)          # MuJoCo >= 3.4
    except TypeError:
        mujoco.mj_fullM(model, out, data.qM)       # older releases


class ComputedTorqueController:
    """tau = M(q) [qdd_ref + Kd (qd_ref - qd) + Kp (q_ref - q)] + h(q, qd)."""

    def __init__(self, model):
        self.dofs = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.lim = np.array([model.actuator(a).ctrlrange[1] for a in ARM_ACTS])
        self.M = np.zeros((model.nv, model.nv))

    def __call__(self, model, data, q_ref, qd_ref, qdd_ref):
        q, qd = data.qpos[self.qadr], data.qvel[self.dofs]
        full_inertia(model, data, self.M)
        M = self.M[np.ix_(self.dofs, self.dofs)]
        v = qdd_ref + KD * (qd_ref - qd) + KP * (q_ref - q)
        tau = M @ v + data.qfrc_bias[self.dofs] - data.qfrc_passive[self.dofs]
        return np.clip(tau, -self.lim, self.lim), q


def plan_task(model, planner):
    """Compute IK way-points and energy-optimal segments for approach, descend, grasp, lift."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obj = data.body("object").xpos.copy()
    obj_mass = model.body("object").mass[0]

    ik = lambda p, q_init: solve_ik(model, "tcp", ARM_JOINTS, p, q_init)
    q_pre = ik(obj + [0, 0, 0.12], Q_HOME)
    q_grasp = ik(obj + [0, 0, 0.005], q_pre)
    q_lift = ik(obj + [0, 0, 0.20], q_grasp)

    plan = []          # (name, trajectory, gripper command, J_opt, J_minjerk)
    #  name        start    goal     T guess  corridor  payload   gripper
    for name, qa, qb, T, corridor, payload, grip in [
        ("approach", Q_HOME, q_pre, 2.0, False, 0.0, GRIP_OPEN),       # free-space move
        ("descend", q_pre, q_grasp, 1.0, True, 0.0, GRIP_OPEN),        # straight down onto the box
        ("lift", q_grasp, q_lift, 1.5, True, obj_mass, GRIP_CLOSED),   # straight up, carrying the box
    ]:
        best, base, J, Jb = planner.plan(qa, qb, T, free_time=True, corridor=corridor,
                                         payload_body="palm", payload_mass=payload)
        print(f"  {name:9s} T* = {best.T:4.2f} s   effort: optimal {J:7.2f}  vs min-jerk {Jb:7.2f} N²m²s"
              f"   ({100 * (1 - J / Jb):4.1f}% lower)")
        plan.append((name, best, grip, J, Jb))
        if name == "descend":          # close the gripper and let the grasp settle
            plan.append(("grasp", Trajectory.hold(q_grasp, 0.6), GRIP_CLOSED, None, None))
    plan.append(("hold", Trajectory.hold(q_lift, 1.0), GRIP_CLOSED, None, None))
    return plan


def simulate(model, data, plan, viewer=None, renderer=None, cam=None, fps=30):
    ctrl = ComputedTorqueController(model)
    arm_act = [model.actuator(a).id for a in ARM_ACTS]
    fing_act = [model.actuator(a).id for a in FINGER_ACTS]

    log = {k: [] for k in ["t", "q", "q_ref", "tau", "obj_z", "phase"]}
    frames, energy = [], {}
    dt = model.opt.timestep
    for phase, (name, traj, grip, _, _) in enumerate(plan):
        t0, e = data.time, 0.0
        while data.time - t0 < traj.T - 1e-9:
            q_ref, qd_ref, qdd_ref = traj.eval(data.time - t0)
            tau, q = ctrl(model, data, q_ref, qd_ref, qdd_ref)
            data.ctrl[arm_act], data.ctrl[fing_act] = tau, grip
            mujoco.mj_step(model, data)
            e += tau @ tau * dt

            log["t"].append(data.time); log["q"].append(q.copy()); log["q_ref"].append(q_ref)
            log["tau"].append(tau); log["obj_z"].append(data.body("object").xpos[2]); log["phase"].append(phase)

            if renderer is not None and len(frames) < data.time * fps:
                renderer.update_scene(data, camera=cam); frames.append(renderer.render())
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0.0, dt - 1e-4))
        energy[name] = e
    return {k: np.array(v) for k, v in log.items()}, energy, frames


def plot(log, plan, energy, path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    t = log["t"]
    for i, j in enumerate(ARM_JOINTS):
        ax[0, 0].plot(t, log["q"][:, i], label=j)
        ax[0, 0].plot(t, log["q_ref"][:, i], "k--", lw=0.8)
        ax[0, 1].plot(t, log["tau"][:, i], label=j)
    ax[0, 0].set(title="Joint angles (dashed = optimal reference)", xlabel="t [s]", ylabel="rad")
    ax[0, 1].set(title="Applied joint torques", xlabel="t [s]", ylabel="N·m")
    ax[1, 0].plot(t, log["obj_z"] * 100, color="tab:blue")
    ax[1, 0].set(title="Object height", xlabel="t [s]", ylabel="cm")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        for k in np.flatnonzero(np.diff(log["phase"])):
            a.axvline(t[k], color="gray", lw=0.6, ls=":")
    ax[0, 0].legend(fontsize=8); ax[0, 1].legend(fontsize=8)

    segs = [p for p in plan if p[3] is not None]
    x = np.arange(len(segs))
    ax[1, 1].bar(x - 0.2, [p[4] for p in segs], 0.4, label="min-jerk (same T)", color="tab:gray")
    ax[1, 1].bar(x + 0.2, [p[3] for p in segs], 0.4, label="energy-optimal", color="tab:orange")
    ax[1, 1].set_xticks(x, [p[0] for p in segs])
    ax[1, 1].set(title=r"Effort $\int \tau^T\tau\,dt$ (model)", ylabel="N²m²s"); ax[1, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="run without the interactive viewer")
    ap.add_argument("--video", action="store_true", help="save results/pick.mp4 and pick.gif (needs imageio[ffmpeg])")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    model = mujoco.MjModel.from_xml_path(XML)
    planner = EnergyOptimalPlanner(XML, ARM_JOINTS, ARM_ACTS)
    print("Planning energy-optimal segments...")
    plan = plan_task(model, planner)

    data = mujoco.MjData(model)
    data.qpos[[model.joint(j).qposadr[0] for j in ARM_JOINTS]] = Q_HOME
    mujoco.mj_forward(model, data)

    renderer, cam = None, None
    if args.video:
        renderer = mujoco.Renderer(model, 720, 960)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.15, 0.05, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 1.7, 135, -18

    if args.headless:
        log, energy, frames = simulate(model, data, plan, renderer=renderer, cam=cam)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            log, energy, frames = simulate(model, data, plan, viewer=viewer, renderer=renderer, cam=cam)

    lifted = log["obj_z"][-1] - log["obj_z"][0]
    err = np.abs(log["q"] - log["q_ref"]).max()
    print(f"\nSimulation: object lifted by {lifted * 100:.1f} cm -> {'SUCCESS' if lifted > 0.1 else 'FAILED'}")
    print(f"Max joint tracking error: {np.degrees(err):.2f} deg")
    print("Measured effort per phase [N²m²s]: " + ", ".join(f"{k} {v:.2f}" for k, v in energy.items()))
    plot(log, plan, energy, "results/pick_results.png")
    print("Saved results/pick_results.png")
    if frames:
        import imageio
        imageio.mimsave("results/pick.mp4", frames, fps=30)
        imageio.mimsave("results/pick.gif", [f[::2, ::2] for f in frames], duration=1000 / 30, loop=0)
        print("Saved results/pick.mp4 and results/pick.gif")
