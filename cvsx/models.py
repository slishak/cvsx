from dataclasses import fields
from typing import Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from cvsx import components as c
from cvsx import cardiac_drivers as drv
from cvsx import parameters as p


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
    v_tot: float
    p_pl: float
    p_pl_affects_pu_and_pa: bool = True
    volume_ratios: bool = False
    smooth_valves: bool = True

    def __init__(
        self,
        parameter_source: str = "smith",
        p_pl_affects_pu_and_pa: bool = True,
        volume_ratios: bool = False,
        cd: Optional[drv.CardiacDriverBase] = None,
    ):
        params = p.cvs_parameters[parameter_source]
        self.parameterise(params, cd=cd)

        self.p_pl_affects_pu_and_pa = p_pl_affects_pu_and_pa
        self.volume_ratios = volume_ratios

        if volume_ratios:
            raise NotImplementedError

    def parameterise(self, params: dict, cd: Optional[drv.CardiacDriverBase] = None):
        if cd is None:
            cd = drv.SimpleCardiacDriver()
        for field in fields(self):
            if field.name == "cd":
                self.cd = cd
                continue

            try:
                field_params = params[field.name]
            except KeyError:
                continue

            if isinstance(field_params, dict):
                setattr(self, field.name, field.type(**field_params))
            else:
                setattr(self, field.name, field.type(field_params))

    def __call__(
        self,
        t: jnp.ndarray,
        states: dict,
        args: tuple[diffrax.AbstractNonlinearSolver],
        all_outputs=False,
    ) -> dict:
        solver = args[0]
        e_t = self.cd(t)
        p_v = self.pressures_volumes(e_t, states, solver)
        flow_rates = self.flow_rates(states, p_v)
        derivatives = self.derivatives(flow_rates, p_v)

        # jax.debug.print("{t}", t=t)
        # jax.debug.print("{t}\n{states}\n{derivatives}", t=t, states=states, derivatives=derivatives)

        if all_outputs:
            out = p_v | flow_rates
            out.update({f"d{key}_dt": val for key, val in derivatives.items()})
            out["e_t"] = e_t
            return out

        return derivatives

    def pressures_volumes(self, e_t, states, solver):
        v_spt = self.solve_v_spt(states["v_lv"], states["v_rv"], e_t, solver)

        v_lvf = states["v_lv"] - v_spt
        v_rvf = states["v_rv"] + v_spt
        p_lvf = self.lvf.p(v_lvf, e_t)
        p_rvf = self.rvf.p(v_rvf, e_t)

        v_pcd = states["v_lv"] + states["v_rv"]
        p_pcd = self.pcd.p_ed(v_pcd)
        p_peri = p_pcd + self.p_pl

        p_lv = p_lvf + p_peri
        p_rv = p_rvf + p_peri

        p_pa = self.pa.p_es(states["v_pa"])
        p_pu = self.pu.p_es(states["v_pu"])
        p_ao = self.ao.p_es(states["v_ao"])
        p_vc = self.vc.p_es(states["v_vc"])

        if self.p_pl_affects_pu_and_pa:
            p_pa = p_pa + self.p_pl
            p_pu = p_pu + self.p_pl

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

    def flow_rates(self, states: dict, p_v: dict) -> dict:
        return {
            "q_mt": self.mt.flow_rate(p_v["p_pu"], p_v["p_lv"]),
            "q_av": self.av.flow_rate(p_v["p_lv"], p_v["p_ao"]),
            "q_tc": self.tc.flow_rate(p_v["p_vc"], p_v["p_rv"]),
            "q_pv": self.pv.flow_rate(p_v["p_rv"], p_v["p_pa"]),
            "q_pul": self.pul.flow_rate(p_v["p_pa"], p_v["p_pu"]),
            "q_sys": self.sys.flow_rate(p_v["p_ao"], p_v["p_vc"]),
        }

    def derivatives(self, flow_rates: dict, p_v: dict) -> dict:
        if self.volume_ratios:
            deriv_scale = self.v_tot
        else:
            deriv_scale = 1.0

        derivatives = {
            "v_pa": (flow_rates["q_pv"] - flow_rates["q_pul"]) / deriv_scale,
            "v_pu": (flow_rates["q_pul"] - flow_rates["q_mt"]) / deriv_scale,
            "v_lv": (flow_rates["q_mt"] - flow_rates["q_av"]) / deriv_scale,
            "v_ao": (flow_rates["q_av"] - flow_rates["q_sys"]) / deriv_scale,
            "v_vc": (flow_rates["q_sys"] - flow_rates["q_tc"]) / deriv_scale,
            "v_rv": (flow_rates["q_tc"] - flow_rates["q_pv"]) / deriv_scale,
        }

        return derivatives

    def solve_v_spt(self, v_lv, v_rv, e_t, solver):
        sol = lambda v_lv_i, v_rv_i, e_t_i: solver(
            self.v_spt_residual, 0.0, (v_lv_i, v_rv_i, e_t_i)
        ).root
        sol_v = jax.vmap(sol, (0, 0, 0), 0)
        v_lv_1d = jnp.atleast_1d(v_lv)
        v_rv_1d = jnp.atleast_1d(v_rv)
        e_t_1d = jnp.atleast_1d(e_t)
        v_spt = sol_v(v_lv_1d, v_rv_1d, e_t_1d)

        v_spt = jnp.reshape(v_spt, v_lv.shape)

        return v_spt

        # solution = solver(self.v_spt_residual, v_spt_guess, (v_lv, v_rv, e_t))
        # return solution.root

    def v_spt_residual(self, v_spt, args):
        v_lv, v_rv, e_t = args
        v_lvf = v_lv - v_spt
        v_rvf = v_rv + v_spt

        res = self.spt.p(v_spt, e_t) - self.lvf.p(v_lvf, e_t) + self.rvf.p(v_rvf, e_t)

        return res


class InertialSmithCVS(SmithCVS):
    mt: c.InertialValve
    tc: c.InertialValve
    av: c.InertialValve
    pv: c.InertialValve

    def __init__(self, parameter_source: str = "revie", *args, **kwargs):
        super().__init__(parameter_source, *args, **kwargs)

    def derivatives(self, flow_rates: dict, p_v: dict) -> dict:
        derivatives = super().derivatives(flow_rates, p_v)
        derivatives["q_mt"] = self.mt.flow_rate_deriv(p_v["p_pu"], p_v["p_lv"], flow_rates["q_mt"])
        derivatives["q_av"] = self.av.flow_rate_deriv(p_v["p_lv"], p_v["p_ao"], flow_rates["q_av"])
        derivatives["q_tc"] = self.tc.flow_rate_deriv(p_v["p_vc"], p_v["p_rv"], flow_rates["q_tc"])
        derivatives["q_pv"] = self.pv.flow_rate_deriv(p_v["p_rv"], p_v["p_pa"], flow_rates["q_pv"])

        return derivatives

    def flow_rates(self, states: dict, p_v: dict) -> dict:
        return {
            "q_mt": jnp.maximum(states["q_mt"], 0.0),
            "q_av": jnp.maximum(states["q_av"], 0.0),
            "q_tc": jnp.maximum(states["q_tc"], 0.0),
            "q_pv": jnp.maximum(states["q_pv"], 0.0),
            "q_pul": self.pul.flow_rate(p_v["p_pa"], p_v["p_pu"]),
            "q_sys": self.sys.flow_rate(p_v["p_ao"], p_v["p_vc"]),
        }


class SmoothInertialSmithCVS(InertialSmithCVS):
    mt: c.SmoothInertialValve
    tc: c.SmoothInertialValve
    av: c.SmoothInertialValve
    pv: c.SmoothInertialValve

    def flow_rates(self, states: dict, p_v: dict) -> dict:
        return {
            "q_mt": states["q_mt"],
            "q_av": states["q_av"],
            "q_tc": states["q_tc"],
            "q_pv": states["q_pv"],
            "q_pul": self.pul.flow_rate(p_v["p_pa"], p_v["p_pu"]),
            "q_sys": self.sys.flow_rate(p_v["p_ao"], p_v["p_vc"]),
        }
