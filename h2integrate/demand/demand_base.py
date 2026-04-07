from attrs import field, define

from h2integrate.core.utilities import BaseConfig
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


@define(kw_only=True)
class DemandComponentBaseConfig(BaseConfig):
    """Configuration for defining an open-loop demand profile.

    This configuration object specifies the commodity being controlled and the
    demand profile that should be met by downstream components.

    Attributes:
        commodity (str): Name of the commodity being controlled
            (e.g., "hydrogen"). Converted to lowercase and stripped of whitespace.
        commodity_rate_units (str): Units of the commodity (e.g., "kg/h").
        demand_profile (int | float | list): Demand values for each timestep, in
            the same units as `commodity_rate_units`. May be a scalar for constant
            demand or a list/array for time-varying demand.
    """

    commodity: str = field(converter=str.strip)
    commodity_rate_units: str = field(converter=str.strip)
    demand_profile: int | float | list = field()


class DemandComponentBase(PerformanceModelBaseClass):
    """Base OpenMDAO component for open-loop demand tracking.

    This component defines the interfaces required for open-loop demand
    controllers, including inputs for demand, supplied commodity, and outputs
    tracking unmet demand, unused production, and total unmet demand.
    Subclasses must implement the :meth:`compute` method to define the
    controller behavior.
    """

    def setup(self):
        """Define inputs and outputs for demand control.

        Creates time-series inputs and outputs for commodity demand, supply,
        unmet demand, unused commodity, and total unmet demand. Shapes and units
        are determined by the plant configuration and controller configuration.

        Raises:
            KeyError: If required configuration keys are missing from
                ``plant_config`` or ``tech_config``.
        """
        self.commodity = self.config.commodity
        self.commodity_rate_units = self.config.commodity_rate_units
        self.commodity_amount_units = getattr(
            self.config, "commodity_amount_units", f"({self.config.commodity_rate_units})*h"
        )

        super().setup()

        self.add_input(
            f"{self.commodity}_demand",
            val=self.config.demand_profile,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"Demand profile of {self.commodity}",
        )

        self.add_input(
            f"{self.commodity}_in",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"Amount of {self.commodity} demand that has already been supplied",
        )

        self.add_output(
            f"unmet_{self.commodity}_demand_out",
            val=self.config.demand_profile,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"Remaining demand profile of {self.commodity}",
        )

        self.add_output(
            f"unused_{self.commodity}_out",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"Excess production of {self.commodity}",
        )

    def compute():
        """This method must be implemented by subclasses to define the
        controller.

        Raises:
            NotImplementedError: Always, unless implemented in a subclass.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")
