from attrs import field, define

from h2integrate.core.utilities import CostModelBaseConfig, merge_shared_inputs
from h2integrate.core.validators import gt_zero, range_val
from h2integrate.core.model_baseclasses import CostModelBaseClass


@define
class ATBBatteryCostConfig(CostModelBaseConfig):
    """Configuration class for the ATBBatteryCostModel with costs based on storage
    capacity and charge rate. More information on ATB methodology and representative
    battery technologies can be found
    `here <https://atb.nrel.gov/electricity/2024/utility-scale_battery_storage>`_
    Reference cost values can be found on the `Utility-Scale Battery Storage`,
    `Commercial Battery Storage`, and `Residential Battery Storage` sheets of the
    `NREL ATB workbook <https://atb.nrel.gov/electricity/2024/data>`_.

    Attributes:
        energy_capex (float|int): battery energy capital cost in $/kWh
        power_capex (float|int): battery power capital cost in $/kW
        opex_fraction (float): annual operating cost as a fraction of the total system cost.
        cost_year (int): dollar year corresponding to input costs
    """

    energy_capex: float | int = field(validator=gt_zero)
    power_capex: float | int = field(validator=gt_zero)
    opex_fraction: float = field(validator=range_val(0, 1))


class ATBBatteryCostModel(CostModelBaseClass):
    def setup(self):
        self.config = ATBBatteryCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )

        super().setup()

        self.add_input("charge_rate", val=0.0, units="kW", desc="Battery charge/discharge rate")
        self.add_input("storage_capacity", val=0.0, units="kW*h", desc="Battery storage capacity")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        storage_duration_hrs = inputs["storage_capacity"][0] / inputs["charge_rate"][0]

        total_system_cost = (
            storage_duration_hrs * self.config.energy_capex
        ) + self.config.power_capex
        capex = total_system_cost * inputs["charge_rate"][0]
        opex = self.config.opex_fraction * capex
        outputs["CapEx"] = capex
        outputs["OpEx"] = opex
