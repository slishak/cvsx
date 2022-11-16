import pytest
import numpy as np
import jax.numpy as jnp

from cvsx.unit_conversions import Converter, convert


def test_conversion_pa_to_kpa():
    pa = 1000
    kpa = convert(pa, "Pa", "kPa")
    assert kpa == 1


def test_default_to_l():
    ml = 1000
    l = convert(ml, to="l")
    assert l == 1


def test_default_from_l():
    l = 1
    ml = convert(l, "l")
    assert ml == 1000


@pytest.mark.parametrize("quantity", [q.default for q in Converter().quantities])
def test_no_conversion_with_default(quantity):
    value = 1
    l = convert(value, quantity)
    assert value is l


def test_exception_if_no_units():
    val = 1
    with pytest.raises(ValueError):
        convert(val)


def test_exception_if_invalid_units():
    val = 1
    with pytest.raises(ValueError):
        convert(val, "k")


def test_exception_if_incompatible_units():
    val = 1
    with pytest.raises(ValueError):
        convert(val, "mmHg", to="s")


def test_modify_default_units():
    converter = Converter(default_pressure="kPa")
    pa = 1000
    kpa = converter.convert(pa, "Pa")
    assert kpa == 1


def test_numpy_conversion():
    kpa = np.array([1.0, 1.5, 2.0])
    pa = convert(kpa, "kPa", "Pa")
    assert isinstance(pa, np.ndarray)
    assert np.allclose(pa, kpa * 1000)


def test_jnp_conversion():
    kpa = jnp.array([1.0, 1.5, 2.0])
    pa = convert(kpa, "kPa", "Pa")
    assert isinstance(pa, jnp.ndarray)
    assert jnp.allclose(pa, kpa * 1000)
