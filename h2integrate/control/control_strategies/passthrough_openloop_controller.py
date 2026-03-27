import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs


@define(kw_only=True)
class PassThroughOpenLoopControllerConfig(BaseConfig):
    commodity: str = field()
    commodity_rate_units: str = field()
    demand_profile: int | float | list = field(default=0.0)


class PassThroughOpenLoopController(om.ExplicitComponent):
    """
    A simple pass-through controller for open-loop systems.

    This controller directly passes the input commodity flow to the output without any
    modifications. It is useful for testing, as a placeholder for more complex controllers,
    and for maintaining consistency between controlled and uncontrolled frameworks as this
    'controller' does not alter the system output in any way.
    """

    def initialize(self):
        """
        Declare options for the component. See "Attributes" section in class doc strings for
        details.
        """

        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        self.config = PassThroughOpenLoopControllerConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "control"),
            additional_cls_name=self.__class__.__name__,
        )

        n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])

        self.add_input(
            f"{self.config.commodity}_in",
            val=0.0,
            shape=n_timesteps,
            units=self.config.commodity_rate_units,
            desc=f"{self.config.commodity} input timeseries from production to storage",
        )

        self.add_input(
            f"{self.config.commodity}_demand",
            val=self.config.demand_profile,
            shape=n_timesteps,
            units=self.config.commodity_rate_units,
            desc=f"{self.config.commodity} demand",
        )

        self.add_output(
            f"{self.config.commodity}_set_point",
            copy_shape=f"{self.config.commodity}_in",
            units=self.config.commodity_rate_units,
            desc=f"{self.config.commodity} output timeseries from plant after storage",
        )

    def compute(self, inputs, outputs):
        """
        Pass through input to output flows.

        Args:
            inputs (dict): Dictionary of input values.
                - {commodity}_in: Input commodity flow.
            outputs (dict): Dictionary of output values.
                - {commodity}_out: Output commodity flow, equal to the input flow.
        """

        if np.sum(inputs[f"{self.config.commodity}_demand"]) > 0:
            commodity_demand = inputs[f"{self.config.commodity}_demand"]
        else:
            # If the commodity_demand is zero, use the average
            # commodity_in as the demand
            commodity_demand = np.mean(inputs[f"{self.config.commodity}_in"]) * np.ones(
                len(inputs[f"{self.config.commodity}_demand"])
            )

        # Assign the input to the output
        outputs[f"{self.config.commodity}_set_point"] = (
            commodity_demand - inputs[f"{self.config.commodity}_in"]
        )

    def setup_partials(self):
        """
        Declare partial derivatives as unity throughout the design space.

        This method specifies that the derivative of the output with respect to the input is
        always 1.0, consistent with the pass-through behavior.

        Note:
        This method is not currently used and isn't strictly needed if you're creating other
        controllers; it is included as a nod towards potential future development enabling
        more derivative information passing.
        """

        # Get the size of the input/output array
        size = self._get_var_meta(f"{self.config.commodity}_in", "size")

        # Declare partials sparsely for all elements as an identity matrix
        # (diagonal elements are 1.0, others are 0.0)
        self.declare_partials(
            of=f"{self.config.commodity}_set_point",
            wrt=f"{self.config.commodity}_in",
            rows=np.arange(size),
            cols=np.arange(size),
            val=np.ones(size),  # Diagonal elements are 1.0
        )
