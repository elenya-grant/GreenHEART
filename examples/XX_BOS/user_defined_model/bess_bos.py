from attrs import field, define

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.model_baseclasses import CostModelBaseClass, CostModelBaseConfig


try:
    from GenStorBOSSE.pvscm_nrel_bess_model import GenStorBOSSEModel
except ImportError:
    GenStorBOSSEModel = None


n_timesteps = 8760


@define(kw_only=True)
class BESSCostConfig(CostModelBaseConfig):
    max_capacity: float = field()
    max_charge_rate: float = field()
    # how to include the other parameters?


class BESSCostModel(CostModelBaseClass):
    def initialize(self):
        super().initialize()
        if GenStorBOSSEModel is None:
            raise ImportError(
                "The `GenStorBOSSE` package is required to use the cost model. "
                "Install it via:\n"
                "pip install git+https://github.com/dmulash/GenStorBOSSE.git"
            )

    def setup(self):
        self.config = BESSCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost")
        )

        super().setup()

        self.add_input(
            "max_capacity",
            val=0.0,
            units="kWh",
            desc="Maximum energy capacity of the BESS",
        )

        self.add_input(
            "max_charge_rate",
            val=0.0,
            units="kW",
            desc="Maximum charge rate of the BESS",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        pass
