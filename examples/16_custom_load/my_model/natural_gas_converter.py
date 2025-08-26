import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, CostModelBaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import CostModelBaseClass


n_timesteps = 8760


@define
class NaturalGasPerformanceConfig(BaseConfig):
    rated_capacity_MW: float = field()
    heat_rate_mmbtu_per_mwh: float = field(default=12)
    ng_avail_mmbtu: list | float = field(default=[0.0])
    # operation_mode: str = field(
    #     default = 'supply',
    #     converter=(str.strip, str.lower),
    #     validator=contains(['demand','supply'])
    #     )


class NaturalGasPerformanceModel(om.ExplicitComponent):
    """
    Base class for natural gas plant performance models.

    This base class defines the common interface for natural gas combustion
    turbine (NGCT) and natural gas combined cycle (NGCC) performance models.
    """

    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = NaturalGasPerformanceConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )

        self.add_input(
            "heat_rate_mmbtu_per_mwh",
            val=self.config.heat_rate_mmbtu_per_mwh,
            units="MMBtu/MW/h",
            desc="Plant heat rate in MMBtu/MWh",
        )

        self.add_input(
            "rated_capacity_electricity",
            val=self.config.rated_capacity_MW,
            units="MW",
            desc="Plant rated capacity in MW",
        )

        self.add_input(
            "electricity_in",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Natural gas output electricity demand",
        )

        self.add_input(
            "natural_gas_in",
            val=self.config.ng_avail_mmbtu,
            shape=n_timesteps,
            units="MMBtu",
            desc="Natural gas input energy",
        )

        self.add_output(
            "natural_gas_consumed",
            val=0.0,
            shape=n_timesteps,
            units="MMBtu",
            desc="Natural gas input energy",
        )

        self.add_output(
            "electricity_out",
            val=0.0,
            shape=n_timesteps,
            units="MW",
            desc="Electricity output from natural gas plant",
        )

    def compute(self, inputs, outputs):
        heat_rate_mmbtu_per_mwh = inputs["heat_rate_mmbtu_per_mwh"][0]
        rated_capacity_MW = inputs["rated_capacity_electricity"]

        rated_capacity_MMBtu = self.config.rated_capacity_MW * heat_rate_mmbtu_per_mwh

        # if self.config.operation_mode == 'demand':
        output_demand_given = True
        input_availability_given = True
        if np.sum(inputs["electricity_in"]) == 0.0:
            output_demand_given = False
        if np.sum(inputs["natural_gas_in"]) == 0.0:
            input_availability_given = False

        electricity_demand = inputs["electricity_in"]
        electricity_supplied = np.where(
            electricity_demand > rated_capacity_MW, rated_capacity_MW, electricity_demand
        )
        natural_gas_required = electricity_supplied * heat_rate_mmbtu_per_mwh

        # if self.config.operation_mode == 'supply':
        natural_gas_available = inputs["natural_gas_in"]
        natural_gas_used = np.where(
            natural_gas_available > rated_capacity_MMBtu,
            rated_capacity_MMBtu,
            natural_gas_available,
        )
        max_electricity_out = natural_gas_used / heat_rate_mmbtu_per_mwh

        if not output_demand_given and input_availability_given:
            # gave natural gas (feedstock) availability, not electricity (output) demand
            # provided available input, did not provide demand
            outputs["electricity_out"] = max_electricity_out
            outputs["natural_gas_consumed"] = natural_gas_used

        if not output_demand_given and input_availability_given:
            # gave electricity (output) demand, not natural gas (feedstock) availability,
            # provided available input, did not provide demand
            outputs["electricity_out"] = electricity_supplied
            outputs["natural_gas_consumed"] = natural_gas_required

        if output_demand_given and input_availability_given:
            # gave natural gas (feedstock) availability AND gave electricity (output) demand
            electricity_prod = np.minimum.reduce([electricity_supplied, max_electricity_out])
            ng_required = electricity_prod * heat_rate_mmbtu_per_mwh
            outputs["electricity_out"] = electricity_prod
            outputs["natural_gas_consumed"] = ng_required


@define
class NaturalGasCostConfig(CostModelBaseConfig):
    capex_per_kW: float = field()
    opex_per_kW: float = field()
    rated_capacity_MW: float = field(default=10)


class NaturalGasCost(CostModelBaseClass):
    def setup(self):
        self.config = NaturalGasCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )

        super().setup()
        self.add_input(
            "rated_capacity_electricity",
            val=self.config.rated_capacity_MW,
            units="MW",
            desc="Plant rated capacity in MW",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        capex = inputs["rated_capacity_electricity"][0] * 1e3 * self.config.capex_per_kW
        opex = inputs["rated_capacity_electricity"][0] * 1e3 * self.config.opex_per_kW

        outputs["CapEx"] = capex
        outputs["OpEx"] = opex
