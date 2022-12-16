from dataclasses import fields
from typing import Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from cvsx import components as c
from cvsx import cardiac_drivers as drv

# from cvsx import parameters as p
from cvsx import respiratory as resp


class SmithCVS(eqx.Module):
    mt: c.Valve
    tc: c.Valve
    av: c.Valve
    pv: c.Valve
    pul: c.BloodVessel
    sys: c.BloodVessel
    lvf: c.PressureVolume
    rvf: c.PressureVolume
    spt: c.PressureVolume
    pcd: c.PressureVolume
    vc: c.PressureVolume
    pa: c.PressureVolume
    pu: c.PressureVolume
    ao: c.PressureVolume
    cd: drv.CardiacDriverBase
    p_pl: float
    p_pl_affects_pu_and_pa: bool = True
    p_pl_is_input: bool = False
    v_spt_method: str = "solver"

    def __call__(
        self,
        t: jnp.ndarray,
        states: dict,
        args: (
            tuple[diffrax.AbstractNonlinearSolver]
            | tuple[diffrax.AbstractNonlinearSolver, jnp.ndarray]
        ),
        return_outputs: bool = False,
    ) -> dict:

        solver = args[0]
        if self.p_pl_is_input:
            p_pl = args[1]
        else:
            p_pl = self.p_pl

        if self.cd.dynamic:
            e_t, ds_dt = self.cd(t, states["s"])
        else:
            e_t = self.cd(t)

        p_v = self.pressures_volumes(t, e_t, states, solver, p_pl)
        flow_rates = self.flow_rates(t, states, p_v)
        derivatives = self.derivatives(t, flow_rates, p_v)

        if self.cd.dynamic:
            derivatives["s"] = ds_dt

        # jax.debug.print("{t}", t=t)
        # jax.debug.print("{t}\n{states}\n{derivatives}", t=t, states=states, derivatives=derivatives)

        if not return_outputs:
            return derivatives

        outputs = p_v | flow_rates
        outputs["e_t"] = e_t
        return derivatives, outputs

    @property
    def connections(self) -> dict[str, tuple[c.BloodVessel, str, str]]:
        return {
            "q_mt": (self.mt, "p_pu", "p_lv"),
            "q_av": (self.av, "p_lv", "p_ao"),
            "q_sys": (self.sys, "p_ao", "p_vc"),
            "q_tc": (self.tc, "p_vc", "p_rv"),
            "q_pv": (self.pv, "p_rv", "p_pa"),
            "q_pul": (self.pul, "p_pa", "p_pu"),
        }

    def pressures_volumes(self, t, e_t, states, solver, p_pl):

        v_spt = self.solve_v_spt(t, states["v_lv"], states["v_rv"], e_t, solver)

        v_lvf = states["v_lv"] - v_spt
        v_rvf = states["v_rv"] + v_spt
        p_lvf = self.lvf.p(t, v_lvf, e_t)
        p_rvf = self.rvf.p(t, v_rvf, e_t)

        v_pcd = states["v_lv"] + states["v_rv"]
        p_pcd = self.pcd.p_ed(t, v_pcd)
        p_peri = p_pcd + p_pl

        p_lv = p_lvf + p_peri
        p_rv = p_rvf + p_peri

        p_pa = self.pa.p_es(t, states["v_pa"])
        p_pu = self.pu.p_es(t, states["v_pu"])
        p_ao = self.ao.p_es(t, states["v_ao"])
        p_vc = self.vc.p_es(t, states["v_vc"])

        if self.p_pl_affects_pu_and_pa:
            p_pa = p_pa + p_pl
            p_pu = p_pu + p_pl

        p_v = {
            "v_pcd": v_pcd,
            "p_pcd": p_pcd,
            "p_peri": p_peri,
            "v_spt": v_spt,
            "v_lvf": v_lvf,
            "v_rvf": v_rvf,
            "p_lvf": p_lvf,
            "p_rvf": p_rvf,
            "p_lv": p_lv,
            "p_rv": p_rv,
            "p_pa": p_pa,
            "p_pu": p_pu,
            "p_ao": p_ao,
            "p_vc": p_vc,
        }

        return p_v

    @staticmethod
    def _compute_flow_rate(
        t: jnp.ndarray,
        states: dict,
        p_v: dict,
        valve: c.Valve,
        name: str,
        p_upstream: str,
        p_downstream: str,
    ) -> jnp.ndarray:
        if valve.inertial:
            if valve.allow_reverse_flow:
                return states[name]
            else:
                return jnp.maximum(states[name], 0.0)
        else:
            return valve.flow_rate(t, p_v[p_upstream], p_v[p_downstream])

    def flow_rates(self, t: jnp.ndarray, states: dict, p_v: dict) -> dict:

        flow_rates = {
            name: self._compute_flow_rate(t, states, p_v, vessel, name, p_upstream, p_downstream)
            for name, (vessel, p_upstream, p_downstream) in self.connections.items()
        }

        return flow_rates

    def derivatives(self, t: jnp.ndarray, flow_rates: dict, p_v: dict) -> dict:
        derivatives = {
            "v_pa": flow_rates["q_pv"] - flow_rates["q_pul"],
            "v_pu": flow_rates["q_pul"] - flow_rates["q_mt"],
            "v_lv": flow_rates["q_mt"] - flow_rates["q_av"],
            "v_ao": flow_rates["q_av"] - flow_rates["q_sys"],
            "v_vc": flow_rates["q_sys"] - flow_rates["q_tc"],
            "v_rv": flow_rates["q_tc"] - flow_rates["q_pv"],
        }

        for name, (vessel, p_upstream, p_downstream) in self.connections.items():
            if vessel.inertial:
                derivatives[name] = vessel.flow_rate_deriv(
                    t, p_v[p_upstream], p_v[p_downstream], flow_rates[name]
            )

        return derivatives

    def solve_v_spt(self, t, v_lv, v_rv, e_t, solver):
        if self.v_spt_method == "solver":

            def func(v_lv_i, v_rv_i, t_i, e_t_i):
                solution = solver(self.v_spt_residual, 0.0, (v_lv_i, v_rv_i, t_i, e_t_i))
                root = jnp.where(solution.result == 0, solution.root, jnp.nan)
                return root

            sol_v = jax.vmap(func, (0, 0, 0, 0), 0)
            v_lv_1d = jnp.atleast_1d(v_lv)
            v_rv_1d = jnp.atleast_1d(v_rv)
            t_1d = jnp.atleast_1d(t)
            e_t_1d = jnp.atleast_1d(e_t)
            v_spt = sol_v(v_lv_1d, v_rv_1d, t_1d, e_t_1d)

            v_spt = jnp.reshape(v_spt, v_lv.shape)
        elif self.v_spt_method == "jallon":
            # Linearisation from Jallon 2009
            # fmt: off
            num = e_t * (
                self.lvf.p_es(t, v_lv) - self.rvf.p_es(t, v_rv) + self.spt.e * self.spt.v_d
            ) + (1 - e_t) * (
                self.lvf.p_ed_linear(t, v_lv)
                - self.rvf.p_ed_linear(t, v_rv)
                + self.spt.lam * self.spt.p_0 * self.spt.v_0
            )
            den = e_t * (
                self.lvf.e + self.rvf.e + self.spt.e
            ) + (1 - e_t) * (
                self.lvf.lam * self.lvf.p_0
                + self.rvf.lam * self.rvf.p_0
                + self.spt.lam * self.spt.p_0
            )
            # fmt: on
            v_spt = num / den
        elif self.v_spt_method == "off":
            v_spt = jnp.zeros_like(v_lv)
        else:
            raise NotImplementedError(self.v_spt_method)

        return v_spt

        # solution = solver(self.v_spt_residual, v_spt_guess, (v_lv, v_rv, e_t))
        # return solution.root

    def v_spt_residual(self, v_spt, args):
        v_lv, v_rv, t, e_t = args
        v_lvf = v_lv - v_spt
        v_rvf = v_rv + v_spt

        res = self.spt.p(t, v_spt, e_t) - self.lvf.p(t, v_lvf, e_t) + self.rvf.p(t, v_rvf, e_t)

        return res


class JallonHeartLungs(eqx.Module):
    cvs: SmithCVS
    resp_sys: resp.PassiveRespiratorySystem
    resp_pattern: resp.RespiratoryPatternGenerator

    def __call__(
        self,
        t: jnp.ndarray,
        states: dict,
        args: tuple[diffrax.AbstractNonlinearSolver],
        return_outputs: bool = False,
    ) -> dict:
        solver = args[0]
        resp_deriv, resp_outputs = self.resp_sys(
            t,
            states,
            return_outputs=True,
        )
        cvs_deriv, cvs_outputs = self.cvs(
            t,
            states,
            (solver, resp_outputs["p_pl"]),
            return_outputs=True,
        )
        resp_pattern_derivs = self.resp_pattern(
            t,
            states,
            (resp_deriv["v_alv"],),
        )

        all_derivs = resp_deriv | cvs_deriv | resp_pattern_derivs

        if not return_outputs:
            return all_derivs

        all_outputs = resp_outputs | cvs_outputs

        return all_derivs, all_outputs
