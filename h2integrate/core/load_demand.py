import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig


@define
class DemandProfileModelConfig(BaseConfig):
    """Config class for feedstock.

    Attributes:
        units (str): demand profile units (such as "galUS" or "kg")
        demand (scalar or list):  The load demand in units of `units`.
            If scalar, demand is assumed to be constant for each timestep.
            If list, then must be the the demand for each timestep.
        resource_type (str, optional): name of the demanded resource. Defaults to 'none'.
    """

    demand: list | int | float = field()
    units: str = field()
    resource_type: str = field(default="none")


class DemandPerformanceModelComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)
        self.options.declare("resource_type", types=str, default="")

    def setup(self):
        n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        demand_profile_options = self.options["tech_config"]["model_inputs"][
            "performance_parameters"
        ]

        demand_profile_options.setdefault("resource_type", self.options["resource_type"])
        self.config = DemandProfileModelConfig.from_dict(demand_profile_options)

        self.add_input(
            f"original_{self.config.resource_type}_demand",
            val=self.config.demand,
            shape=(n_timesteps,),
            units=self.config.units,
            desc=f"Full demand profile of {self.config.resource_type}",
        )

        self.add_output(
            f"{self.config.resource_type}_demand",
            val=self.config.demand,
            shape=(n_timesteps,),
            units=self.config.units,
            desc=f"Remaining demand profile of {self.config.resource_type}",
        )

        self.add_output(
            f"{self.config.resource_type}_curtailed",
            val=0.0,
            shape=(n_timesteps,),
            units=self.config.units,
            desc=f"Excess production of {self.config.resource_type}",
        )

        self.add_input(
            f"{self.config.resource_type}_supplied",
            val=0.0,
            shape=(n_timesteps,),
            units=self.config.units,
            desc=f"Amount of {self.config.resource_type} demand that has already been supplied",
        )

    def compute(self, inputs, outputs):
        remaining_demand = (
            inputs[f"original_{self.config.resource_type}_demand"]
            - inputs[f"{self.config.resource_type}_supplied"]
        )
        current_demand = np.where(remaining_demand > 0, remaining_demand, 0)
        curtailed_demand = np.where(remaining_demand < 0, -1 * remaining_demand, 0)
        outputs[f"{self.config.resource_type}_demand"] = current_demand
        outputs[f"{self.config.resource_type}_curtailed"] = curtailed_demand
