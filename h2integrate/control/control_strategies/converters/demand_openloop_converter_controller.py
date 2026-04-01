import numpy as np

from h2integrate.control.control_strategies.converters.openloop_controller_base import (
    ConverterOpenLoopControlBase,
    ConverterOpenLoopControlBaseConfig,
)


class DemandOpenLoopConverterController(ConverterOpenLoopControlBase):
    """Open-loop controller for converting input supply into met demand.

    This controller computes unmet demand, unused (curtailed) production, and
    the resulting commodity output profile based on the incoming supply and an
    externally specified demand profile. It uses simple arithmetic rules:

    * If demand exceeds supplied commodity, the difference is unmet demand.
    * If supply exceeds demand, the excess is unused (curtailed) commodity.
    * Output equals supplied commodity minus curtailed commodity.

    This component relies on configuration provided through the
    ``tech_config`` dictionary, which must define the controller's
    ``control_parameters``.
    """

    def setup(self):
        """Set up the load controller configuration.

        Loads the controller configuration from ``tech_config`` and then calls
        the base class ``setup`` to create inputs/outputs.

        Raises:
            KeyError: If the expected configuration keys are missing from
                ``tech_config``.
        """
        self.config = ConverterOpenLoopControlBaseConfig.from_dict(
            self.options["tech_config"]["model_inputs"]["control_parameters"],
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

    def compute(self, inputs, outputs):
        """Compute unmet demand, unused commodity, and converter output.

        This method compares the demand profile to the supplied commodity for
        each timestep and assigns unmet demand, curtailed production, and
        actual delivered output.

        Args:
            inputs (dict-like): Mapping of input variable names to their
                current values, including:

                    * ``{commodity}_demand``: Demand profile.
                    * ``{commodity}_in``: Supplied commodity.
            outputs (dict-like): Mapping of output variable names where results
                will be written, including:

                    * ``unmet_{commodity}_demand_out``: Unmet demand.
                    * ``unused_{commodity}_out``: Curtailed production.
                    * ``{commodity}_set_point``: Actual output delivered.

        Notes:
            All variables operate on a per-timestep basis and typically have
            array shape ``(n_timesteps,)``.
        """
        remaining_demand = inputs[f"{self.commodity}_demand"] - inputs[f"{self.commodity}_in"]

        # Calculate missed load and curtailed production
        outputs[f"unmet_{self.commodity}_demand_out"] = np.where(
            remaining_demand > 0, remaining_demand, 0
        )
        outputs[f"unused_{self.commodity}_out"] = np.where(
            remaining_demand < 0, -1 * remaining_demand, 0
        )

        # Calculate actual output based on demand met and curtailment
        outputs[f"{self.commodity}_set_point"] = (
            inputs[f"{self.commodity}_in"] - outputs[f"unused_{self.commodity}_out"]
        )

        # Calculate performance model outputs
        outputs[f"{self.commodity}_out"] = (
            inputs[f"{self.commodity}_in"] - outputs[f"unused_{self.commodity}_out"]
        )

        outputs[f"rated_{self.commodity}_production"] = inputs[f"{self.commodity}_demand"].mean()

        outputs[f"total_{self.commodity}_produced"] = np.sum(outputs[f"{self.commodity}_out"]) * (
            self.dt / 3600
        )
        outputs[f"annual_{self.commodity}_produced"] = (
            outputs[f"total_{self.commodity}_produced"] / self.fraction_of_year_simulated
        )

        outputs["capacity_factor"] = (
            outputs[f"{self.commodity}_set_point"].sum() / inputs[f"{self.commodity}_demand"].sum()
        )
