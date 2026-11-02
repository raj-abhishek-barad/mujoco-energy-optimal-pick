"""Energy-optimal point-to-point joint trajectories using MuJoCo inverse dynamics.

Problem (one segment q0 -> qf):

    min_{c, T}   J = integral_0^T  tau(t)^T tau(t) dt          (actuator effort / ~copper losses)
    s.t.         tau = M(q) qdd + C(q, qd) qd + g(q)           (rigid-body dynamics, via mj_inverse)
                 q(0) = q0, q(T) = qf, qd = qdd = 0 at both ends
                 q_min <= q <= q_max,  |tau| <= tau_max          (handled as penalties)
                 optional Cartesian corridor: the tool stays within 1 cm of the vertical line
                 through its start point and keeps pointing down (for descend / lift moves)

Parameterisation (direct method):
    q(s) = q0 + (qf - q0) * p5(s) + sum_k c_k * phi_k(s),   s = t / T in [0, 1]
p5 is the minimum-jerk quintic. Each basis phi_k(s) = 64 s^3 (1-s)^3 s^k vanishes with its first two
derivatives at both ends, so every candidate satisfies the boundary conditions automatically.
The final time T is an optional decision variable (free-final-time problem).
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.optimize import minimize

MIN_JERK = P([0, 0, 0, 10, -15, 6])
BUMP = 64 * P([0, 0, 0, 1, -3, 3, -1])          # 64 s^3 (1 - s)^3


def _basis(n_basis):
    return [BUMP * P([0] * k + [1]) for k in range(n_basis)]


@dataclass
class Trajectory:
    q0: np.ndarray
    qf: np.ndarray
    T: float
    coeffs: np.ndarray          # shape (n_basis, n_joints)

    def __post_init__(self):
        self._polys = [MIN_JERK] + _basis(len(self.coeffs))

    def eval(self, t):
        """Return q, qd, qdd at time(s) t (scalar or array). Holds qf after T."""
        s = np.clip(np.atleast_1d(t) / self.T, 0.0, 1.0)
        W = np.vstack([self.qf - self.q0, self.coeffs])            # (1 + K, n)
        B0 = np.stack([p(s) for p in self._polys], axis=1)
        B1 = np.stack([p.deriv(1)(s) for p in self._polys], axis=1) / self.T
        B2 = np.stack([p.deriv(2)(s) for p in self._polys], axis=1) / self.T**2
        moving = (np.atleast_1d(t) < self.T)[:, None]
        q, qd, qdd = self.q0 + B0 @ W, (B1 @ W) * moving, (B2 @ W) * moving
        return (q[0], qd[0], qdd[0]) if np.isscalar(t) else (q, qd, qdd)

    @staticmethod
    def hold(q, duration):
        return Trajectory(np.array(q), np.array(q), duration, np.zeros((0, len(q))))


class EnergyOptimalPlanner:
    def __init__(self, xml_path, joint_names, actuator_names, tool_site="tcp", n_basis=3, n_samples=40):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT   # free-space planning
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in joint_names])
        self.dofs = np.array([self.model.joint(j).dofadr[0] for j in joint_names])
        lo, hi = self.model.jnt_range[[self.model.joint(j).id for j in joint_names]].T
        self.q_lo, self.q_hi = lo + 0.1, hi - 0.1                     # 0.1 rad safety margin
        self.tau_max = np.abs(np.array([self.model.actuator(a).ctrlrange[1] for a in actuator_names]))
        self.site = self.model.site(tool_site).id
        self.n, self.K, self.N = len(joint_names), n_basis, n_samples

    # ---------- cost evaluation ----------
    def rollout(self, traj, t):
        """Inverse dynamics along a trajectory: tau(t) plus tool position / approach axis."""
        q, qd, qdd = traj.eval(t)
        tau, tool_pos, tool_z = np.empty_like(q), np.empty((len(t), 3)), np.empty((len(t), 3))
        for i in range(len(t)):
            self.data.qpos[self.qadr], self.data.qvel[self.dofs], self.data.qacc[self.dofs] = q[i], qd[i], qdd[i]
            mujoco.mj_inverse(self.model, self.data)          # also runs forward kinematics
            tau[i] = self.data.qfrc_inverse[self.dofs]
            tool_pos[i] = self.data.site_xpos[self.site]
            tool_z[i] = self.data.site_xmat[self.site].reshape(3, 3)[:, 2]
        return q, tau, tool_pos, tool_z

    def effort(self, traj):
        t = np.linspace(0.0, traj.T, self.N)
        tau = self.rollout(traj, t)[1]
        return np.trapezoid(np.sum(tau**2, axis=1), t)

    def _objective(self, x, q0, qf, T_fixed, corridor):
        coeffs = x[: self.K * self.n].reshape(self.K, self.n)
        T = x[-1] if T_fixed is None else T_fixed
        traj = Trajectory(q0, qf, T, coeffs)
        t = np.linspace(0.0, T, self.N)
        q, tau, p, z = self.rollout(traj, t)
        J = np.trapezoid(np.sum(tau**2, axis=1), t)
        # soft constraints: joint limits and actuator saturation
        pen = np.sum(np.maximum(q - self.q_hi, 0) ** 2 + np.maximum(self.q_lo - q, 0) ** 2) * 1e4
        pen += np.sum(np.maximum(np.abs(tau) - self.tau_max, 0) ** 2) * 1e2
        if corridor:   # stay within 1 cm of the vertical line and within ~5 deg of pointing down
            dev = np.linalg.norm(p[:, :2] - p[0, :2], axis=1)
            tilt = 1.0 + z[:, 2]                               # 0 when pointing straight down
            pen += np.sum(np.maximum(dev - 0.01, 0) ** 2) * 1e6 + np.sum(np.maximum(tilt - 0.004, 0) ** 2) * 1e6
        return J + pen

    # ---------- optimisation ----------
    def plan(self, q0, qf, T, free_time=True, T_bounds=(0.8, 4.0), corridor=False,
             payload_body=None, payload_mass=0.0):
        """Return (optimal trajectory, min-jerk baseline at T*, J_opt, J_baseline)."""
        q0, qf = np.asarray(q0, float), np.asarray(qf, float)
        if payload_body is not None:                       # carry the grasped object in the model
            bid = self.model.body(payload_body).id
            self.model.body_mass[bid] += payload_mass

        bounds = [(-1.0, 1.0)] * (self.K * self.n)          # basis amplitudes (rad)
        starts = [np.zeros(self.K * self.n)]
        if free_time:                                        # multi-start in T: the problem is non-convex
            bounds = bounds + [T_bounds]
            starts = [np.append(starts[0], T0) for T0 in (T_bounds[0], T)]
        res = min((minimize(self._objective, x0, args=(q0, qf, None if free_time else T, corridor),
                            method="L-BFGS-B", bounds=bounds, options={"maxiter": 200}) for x0 in starts),
                  key=lambda r: r.fun)

        T_opt = res.x[-1] if free_time else T
        best = Trajectory(q0, qf, T_opt, res.x[: self.K * self.n].reshape(self.K, self.n))
        baseline = Trajectory(q0, qf, T_opt, np.zeros((self.K, self.n)))
        J_opt, J_base = self.effort(best), self.effort(baseline)

        if payload_body is not None:
            self.model.body_mass[bid] -= payload_mass
        return best, baseline, J_opt, J_base
