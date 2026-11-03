import numpy as np

import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig
from h2integrate.core.supported_models import register
from itertools import combinations

def calculate_hourly_correlation_coeff(profile0,profile1,n_timesteps):
    
    x_d = sum(((profile0[i]-np.mean(profile0))**2) for i in range(n_timesteps))
    y_d = sum(((profile1[i]-np.mean(profile1))**2) for i in range(n_timesteps))
    if (x_d*y_d)!=0:
        num = sum((profile0[i]-np.mean(profile0))*(profile1[i]-np.mean(profile1)) for i in range(len(profile0)))
        r = num/((x_d*y_d)**0.5)
        return r
    # invalid denominator, return large value
    return 2.0


@define(kw_only=True)
class CorrelationCompConfig(BaseConfig):
    # default_output_commodity: str = field(default="electricity")
    # techs_to_output_commodities: list[tuple] = field(default=[])
    techs_specific_outputs_profiles: dict = field(default={})
    pass


@register
class CorrelationComponentModel(om.ExplicitComponent):
    _is_steppable = False

    def initialize(self):
        """Declare OpenMDAO component options."""
        super().initialize()
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)
        self.options.declare("techs_to_output_commodities", types=list)

    
    def setup(self):
        corr_params = self.options["plant_config"]["correlation_parameters"]
        self.n_timesteps = int(self.options["plant_config"]["plant"]["simulation"]["n_timesteps"])
        self.config = CorrelationCompConfig.from_dict(corr_params)

        input_profile_names = []

        for tech, commodity in self.options["techs_to_output_commodities"]:
            # H2I would connect {tech}.{commodity}_out to corr_comp.{tech}_{commodity}_out
            commod_profile_name = f"ts_{tech}_{commodity}_out"
            if commod_profile_name not in input_profile_names:
                self.add_input(commod_profile_name, val=0.0, shape=self.n_timesteps, units_by_conn=True)
                input_profile_names.append(commod_profile_name)

            # for tech_output_profile in self.config.techs_specific_outputs_profiles.get(tech, []):
            #     # H2I would connect {tech}.{tech_output_profile} to corr_comp.{tech}_{tech_output_profile}
            #     input_name = f"{tech}_{tech_output_profile}"
            #     if input_name != commod_profile_name:
            #         self.add_input(input_name, val=0.0, shape=self.n_timesteps, units_by_conn=True)
            #         input_profile_names.append(input_name)
        for tech_name, output_profiles in self.config.techs_specific_outputs_profiles.items():
            if tech_name in self.options["tech_config"]:
                for tech_output_profile in output_profiles:
                    input_name = f"ts_{tech_name}_{tech_output_profile}"
                    if input_name not in input_profile_names:
                        self.add_input(input_name, val=0.0, shape=self.n_timesteps, units_by_conn=True)
                        input_profile_names.append(input_name)

        # combinations to get unique combinations of inputs
        combos = list(combinations(sorted(set(input_profile_names)), 2))

        # for in_name1 in input_profile_names:
        #     for in_name2 in input_profile_names:
        for in_name1, in_name2 in combos:
            if in_name1 == in_name2:
                # skip if same name
                # should not happen
                continue
            in1 = in_name1.split("ts_",1)[-1]
            in2 = in_name2.split("ts_",1)[-1]
            output_name = f"corr_{in1}_to_{in2}"
            self.add_output(output_name, val=0.0, shape=1, units="unitless")
        
        self.combos = combos

    def compute(self, inputs, outputs):
        # skip_vars = []

        # for in1 in dict(inputs.items()):
        #     for in2 in dict(inputs.items()):
        # print(f"{len(self.combos)} combos")
        for in1,in2 in self.combos:
                # if in1 == in2:
                #     # skip if same name
                #     continue
            i1 = in1.split("ts_",1)[-1]
            i2 = in2.split("ts_",1)[-1]
            output_name = f"corr_{i1}_to_{i2}"
            # print("here...")
            outputs[output_name] = calculate_hourly_correlation_coeff(inputs[in1], inputs[in2], self.n_timesteps)

            # if output_name in outputs:
                
            # else:
            #     print(f"{output_name} not in outputs")
            #     raise ValueError(f"{output_name} not in outputs")
                # print(f"{output_name} not in outputs")
                # if (in1 not in skip_vars) and (in2 not in skip_vars):
                #     outputs[output_name] = calculate_hourly_correlation_coeff(inputs[in1], inputs[in2], self.n_timesteps)
                # else:
                #     outputs[output_name] = 2.0
                    # np.corrcoef(inputs[in1], inputs[in2])[0][1]
        
