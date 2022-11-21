import jax.numpy as jnp

from cvsx.unit_conversions import convert


smith_2007 = {
    "mt": {
        "r": jnp.array(convert(0.0600, "kPa s/l")),
    },
    "av": {
        "r": jnp.array(convert(1.4000, "kPa s/l")),
    },
    "tc": {
        "r": jnp.array(convert(0.1800, "kPa s/l")),
    },
    "pv": {
        "r": jnp.array(convert(0.4800, "kPa s/l")),
    },
    "pul": {
        "r": jnp.array(convert(19.0, "kPa s/l")),
    },
    "sys": {
        "r": jnp.array(convert(140.0, "kPa s/l")),
    },
    "lvf": {
        "e": jnp.array(convert(454.0, "kPa/l")),
        "v_d": jnp.array(convert(0.005, "l")),
        "v_0": jnp.array(convert(0.005, "l")),
        "lam": jnp.array(convert(15.0, "1/l")),
        "p_0": jnp.array(convert(0.17, "kPa")),
    },
    "rvf": {
        "e": jnp.array(convert(87.0, "kPa/l")),
        "v_d": jnp.array(convert(0.005, "l")),
        "v_0": jnp.array(convert(0.005, "l")),
        "lam": jnp.array(convert(15.0, "1/l")),
        "p_0": jnp.array(convert(0.16, "kPa")),
    },
    "spt": {
        "e": jnp.array(convert(6500.0, "kPa/l")),
        "v_d": jnp.array(convert(0.002, "l")),
        "v_0": jnp.array(convert(0.002, "l")),
        "lam": jnp.array(convert(435.0, "1/l")),
        "p_0": jnp.array(convert(0.148, "kPa")),
    },
    "pcd": {
        "e": 0.0,
        "v_d": 0.0,
        "v_0": jnp.array(convert(0.2, "l")),
        "lam": jnp.array(convert(30.0, "1/l")),
        "p_0": jnp.array(convert(0.0667, "kPa")),
    },
    "vc": {
        "e": jnp.array(convert(1.5, "kPa/l")),
        "v_d": jnp.array(convert(2.83, "l")),
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "pa": {
        "e": jnp.array(convert(45.0, "kPa/l")),
        "v_d": jnp.array(convert(0.16, "l")),
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "pu": {
        "e": jnp.array(convert(0.8, "kPa/l")),
        "v_d": jnp.array(convert(0.2, "l")),
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "ao": {
        "e": jnp.array(convert(94.0, "kPa/l")),
        "v_d": jnp.array(convert(0.8, "l")),
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "cd": {
        "a": jnp.array([1.0]),
        "b": jnp.array([80.0]),
        "c": jnp.array([0.375]),
        "hr": jnp.array(80.0),
    },
    "p_pl": jnp.array(convert(-4.0, "mmHg")),
    "v_tot": jnp.array(convert(5.5, "l")),
}

revie_2012 = {
    "mt": {
        "r": jnp.array(convert(0.0158, "mmHg s/ml")),
        "l": jnp.array(convert(7.6967e-5, "mmHg s^2/ml")),
    },
    "tc": {
        "r": jnp.array(convert(0.0237, "mmHg s/ml")),
        "l": jnp.array(convert(8.0093e-5, "mmHg s^2/ml")),
    },
    "av": {
        "r": jnp.array(convert(0.0180, "mmHg s/ml")),
        "l": jnp.array(convert(1.2189e-4, "mmHg s^2/ml")),
    },
    "pv": {
        "r": jnp.array(convert(0.0055, "mmHg s/ml")),
        "l": jnp.array(convert(1.4868e-4, "mmHg s^2/ml")),
    },
    "pul": {
        "r": jnp.array(convert(0.1552, "mmHg s/ml")),
    },
    "sys": {
        "r": jnp.array(convert(1.0889, "mmHg s/ml")),
    },
    "lvf": {
        "e": jnp.array(convert(2.8798, "mmHg/ml")),
        "v_d": 0.0,
        "v_0": 0.0,
        "lam": jnp.array(convert(0.033, "1/ml")),
        "p_0": jnp.array(convert(0.1203, "mmHg")),
    },
    "rvf": {
        "e": jnp.array(convert(0.5850, "mmHg/ml")),
        "v_d": 0.0,
        "v_0": 0.0,
        "lam": jnp.array(convert(0.023, "1/ml")),
        "p_0": jnp.array(convert(0.2157, "mmHg")),
    },
    "spt": {
        "e": jnp.array(convert(48.7540, "mmHg/ml")),
        "v_d": jnp.array(convert(2, "ml")),
        "v_0": jnp.array(convert(2.0, "ml")),
        "lam": jnp.array(convert(0.435, "1/ml")),
        "p_0": jnp.array(convert(1.1101, "mmHg")),
    },
    "pcd": {
        "e": 0.0,
        "v_d": 0.0,
        "v_0": jnp.array(convert(200.0, "ml")),
        "lam": jnp.array(convert(0.030, "1/ml")),
        "p_0": jnp.array(convert(0.5003, "mmHg")),
    },
    "vc": {
        "e": jnp.array(convert(0.0059, "mmHg/ml")),
        "v_d": 0.0,
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "pa": {
        "e": jnp.array(convert(0.3690, "mmHg/ml")),
        "v_d": 0.0,
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "pu": {
        "e": jnp.array(convert(0.0073, "mmHg/ml")),
        "v_d": 0.0,
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "ao": {
        "e": jnp.array(convert(0.6913, "mmHg/ml")),
        "v_d": 0.0,
        "v_0": 0.0,
        "lam": 0.0,
        "p_0": 0.0,
    },
    "p_pl": jnp.array(convert(-4.0, "mmHg")),
    "v_tot": jnp.array(convert(1.5, "l")),  # Only simulates stressed volume?
}


cvs_parameters = {
    "smith": smith_2007,
    "revie": revie_2012,
}


cd_smith = {
    "b": jnp.array(80.0),
    "hr": jnp.array(80.0),
}


cd_chung = {
    "a": jnp.array([0.9556, 0.6249, 0.018]),
    "b": jnp.array([255.4, 225.3, 4225.0]),
    "c": jnp.array([0.306, 0.2026, 0.2491]),
    "hr": 80.0,
}


cd_parameters = {
    "smith": cd_smith,
    "chung": cd_chung,
}
