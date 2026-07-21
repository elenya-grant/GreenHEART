import os

from h2integrate import EXAMPLE_DIR
from h2integrate.core.h2integrate_model import H2IntegrateModel


os.chdir(EXAMPLE_DIR / "35_system_level_control" / "complex_multi_commodity")

##################################
# Create an H2I model with a fixed electricity load demand
# h2i = H2IntegrateModel("top_level_config.yaml")

print("Starting V2 ...")
h2i = H2IntegrateModel("top_level_config_v2.yaml")

h2i.setup()

# Run the model
h2i.run()

print("Ran V2 successfully!")

h2i.model.get_val("system_level_controller.nh3_storage_ammonia_set_point", units="kg/h").min()
h2i.model.get_val("system_level_controller.haber_bosch_ammonia_set_point", units="kg/h").max()
h2i.model.get_val(
    "nh3_storage.ammonia_command_value", units="kg/h"
)  # output from storage controller
h2i.model.get_val("nh3_storage.ammonia_out", units="kg/h").max()  # output from performance model
h2i.model.get_val("nh3_storage.ammonia_in", units="kg/h").max()  # input to storage controller
h2i.model.get_val("nh3_load_demand.unmet_ammonia_demand_out", units="kg/h")

# h2i.model.get_val("system_level_controller.")


# print("Starting V1 ...")
# h2i = H2IntegrateModel("top_level_config.yaml")

# h2i.setup()

# # Run the model
# h2i.run()

# print("Ran V1 successfully!")
# Post-process the results
# h2i.post_process()

# TODO: make even more complex by adding in an ammonia storage and combiner that goes to the demand tech
