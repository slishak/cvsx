from dataclasses import dataclass
from typing import Optional, Union

from scipy import constants


@dataclass
class Unit:
    """Definition of a physical unit."""

    name: str
    value: float


@dataclass
class Quantity:
    """Definition of a quantity."""

    name: str
    units: list[Unit]
    default: str

    def find(self, unit_name: Union[str, None]) -> Unit:
        """Find a unit representing this quantity.

        Parameters
        ----------
        unit_name : Union[str, None]
            Name of Unit. If None, find default unit

        Returns
        -------
        Unit
            Object representing the found unit

        Raises
        ------
        ValueError
            When no unit was found for this quantity
        """
        if unit_name is None:
            unit_name = self.default
        for unit in self.units:
            if unit.name == unit_name:
                return unit
        raise ValueError(f"Unit {unit_name} not found in {self.name}")


class Converter:
    def __init__(
        self,
        default_pressure: str = "mmHg",
        default_volume: str = "ml",
    ):
        """_summary_

        Parameters
        ----------
        default_pressure : str, optional
            Default pressure unit, by default "mmHg"
        default_volume : str, optional
            Default volume unit, by default "l"
        """
        self.quantities = [
            Quantity(
                "pressure",
                [
                    Unit("kPa", constants.kilo),
                    Unit("Pa", 1),
                    Unit("mmHg", constants.mmHg),
                    Unit("bar", constants.bar),
                    Unit("psi", constants.psi),
                    Unit("cmH2O", constants.g * constants.centi),
                ],
                default=default_pressure,
            ),
            Quantity(
                "volume",
                [
                    Unit("l", 1),
                    Unit("ml", constants.milli),
                ],
                default=default_volume,
            ),
            Quantity(
                "flow_rate",
                [
                    Unit("l/s", 1),
                    Unit("ml/s", constants.milli),
                ],
                default=f"{default_volume}/s",
            ),
            Quantity(
                "inverse_volume",
                [
                    Unit("1/l", 1),
                    Unit("1/ml", constants.kilo),
                ],
                default=f"1/{default_volume}",
            ),
            Quantity(
                "elastance",
                [
                    Unit("kPa/l", 1),
                    Unit("mmHg/ml", constants.mmHg),
                    Unit("mmHg/l", constants.mmHg * constants.milli),
                    Unit("cmH2O/l", constants.g * constants.centi),
                ],
                default=f"{default_pressure}/{default_volume}",
            ),
            Quantity(
                "resistance",
                [
                    Unit("kPa s/l", 1),
                    Unit("mmHg s/ml", constants.mmHg),
                    Unit("mmHg s/l", constants.mmHg * constants.milli),
                    Unit("cmH2O s/l", constants.g * constants.centi),
                ],
                default=f"{default_pressure} s/{default_volume}",
            ),
            Quantity(
                "inductance",
                [
                    Unit("kPa s^2/l", 1),
                    Unit("mmHg s^2/ml", constants.mmHg),
                    Unit("mmHg s^2/l", constants.mmHg * constants.milli),
                    Unit("cmH2O s^2/l", constants.g * constants.centi),
                ],
                default=f"{default_pressure} s^2/{default_volume}",
            ),
        ]

    def convert(self, value: float, unit: Optional[str] = None, to: Optional[str] = None) -> float:
        """Convert a value from one unit to another.

        At least one of `unit` and `to` must be set. If they are both set, they
        must both be units representing the same quantity (i.e. there must be a
        dimensionless multiplier to convert from one to the other).

        Parameters
        ----------
        value : float
            Value to convert
        unit : Optional[str], optional
            Units of value, by default None (use global default units)
        to : Optional[str], optional
            Unit to convert to, by default None (use global default units)

        Returns
        -------
        float
            Converted value

        Raises
        ------
        ValueError
            When the conversion from "unit" to "to" was not found.
        """

        if unit is None and to is None:
            raise ValueError("Need to provide at least one unit for conversion")
        for quantity in self.quantities:
            try:
                unit_obj = quantity.find(unit)
                to_obj = quantity.find(to)
            except ValueError:
                continue
            else:
                break
        else:
            raise ValueError(f"No quantity found for conversion {unit} -> {to}")

        if unit_obj is to_obj:
            # No conversion needed
            return value

        conversion_factor = unit_obj.value / to_obj.value
        return value * conversion_factor


convert = Converter().convert
