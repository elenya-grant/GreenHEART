import numpy as np
import openmdao.api as om

from h2integrate.core.supported_models import commodity_to_common_units, commodity_to_model_kwargs
from h2integrate.core.model_mapping_tools import get_commodity_for_technology_name


class ElectricitySumComp(om.ExplicitComponent):
    """
    Component to sum up electricity produced through different technologies.
    """

    def initialize(self):
        self.options.declare("tech_configs", types=dict, desc="Configuration for each technology")

    def setup(self):
        # Add inputs for each electricity producing technology
        for tech in self.options["tech_configs"]:
            if tech in commodity_to_model_kwargs["electricity"]:
                self.add_input(
                    f"electricity_{tech}",
                    shape=8760,
                    val=0.0,
                    units="kW",
                    desc=f"Electricity produced by {tech}",
                )

        # Add output for total electricity produced
        self.add_output(
            "total_electricity_produced",
            val=0.0,
            units="kW*h/year",
            desc="Total electricity produced",
        )

    def compute(self, inputs, outputs):
        # Sum up all electricity streams for technologies in electricity_producing_techs
        outputs["total_electricity_produced"] = np.sum(
            [
                inputs[f"electricity_{tech}"]
                for tech in self.options["tech_configs"]
                if tech in commodity_to_model_kwargs["electricity"]
            ]
        )


class GenericSumComp(om.ExplicitComponent):
    """
    Component to sum up commodity types produced through different technologies.
    """

    def initialize(self):
        self.options.declare("tech_configs", types=dict, desc="Configuration for each technology")
        self.options.declare("commodity", types=str)

    def setup(self):
        commodity = self.commodity = self.options["commodity"]
        if commodity not in commodity_to_model_kwargs:
            raise ValueError(f"Unrecognized commodity type: {commodity}")
        units = self.units = commodity_to_common_units[commodity]
        self.valid_techs = []

        for tech in self.options["tech_configs"]:
            commod = get_commodity_for_technology_name(tech, {})
            if commod[0] == commodity:
                self.add_input(
                    f"{commodity}_{tech}",
                    shape=8760,
                    val=0.0,
                    units=units,
                    desc=f"{commodity} produced by {tech}",
                )
                self.valid_techs.append(tech)

        # Add output for total electricity produced
        if commodity == "electricity":
            units = "kW*h"

        self.add_output(
            f"total_{commodity}_produced",
            val=0.0,
            units=f"{units}/year",
            desc=f"Total {commodity} produced",
        )

    def compute(self, inputs, outputs):
        # Sum up all electricity streams for technologies in electricity_producing_techs
        outputs[f"total_{self.commodity}_produced"] = np.sum(
            [inputs[f"{self.commodity}_{tech}"] for tech in self.valid_techs]
        )
