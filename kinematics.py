"""Damped-least-squares inverse kinematics for a MuJoCo site."""
import mujoco
import numpy as np


def solve_ik(model, site_name, joint_names, target_pos, q_init,
             approach_axis=(0.0, 0.0, -1.0), damping=1e-2, tol=1e-6, max_iters=500):
    """Find arm joint angles that put `site_name` at `target_pos`, with the
    site's local z-axis aligned to `approach_axis` (top-down grasp by default).

    Solves  min ||e(q)||  with  dq = J^T (J J^T + lambda^2 I)^-1 e  (Levenberg-Marquardt step).
    """
    data = mujoco.MjData(model)  # scratch data so the simulation state is untouched
    site_id = model.site(site_name).id
    qadr = np.array([model.joint(j).qposadr[0] for j in joint_names])
    dofs = np.array([model.joint(j).dofadr[0] for j in joint_names])
    lo, hi = model.jnt_range[[model.joint(j).id for j in joint_names]].T
    target_pos = np.asarray(target_pos, dtype=float)
    approach_axis = np.asarray(approach_axis, dtype=float)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    q = np.array(q_init, dtype=float)

    for _ in range(max_iters):
        data.qpos[qadr] = q
        mujoco.mj_fwdPosition(model, data)

        pos_err = target_pos - data.site_xpos[site_id]
        z_axis = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        rot_err = np.cross(z_axis, approach_axis)          # rotation that aligns z_axis with target
        err = np.concatenate([pos_err, rot_err])
        if err @ err < tol**2:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        q = np.clip(q + dq, lo, hi)
    else:
        raise RuntimeError(f"IK did not converge for target {target_pos}, residual {np.linalg.norm(err):.2e}")

    return q
