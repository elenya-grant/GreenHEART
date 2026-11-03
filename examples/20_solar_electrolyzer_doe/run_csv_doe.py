"""Minimal working example from the parameter sweep user guide docs."""
# import sys

# print("\n\n")

# for k in sys.path:
#     print(k)

# print("\n\n")
import PySAM
from mpi4py import MPI

from h2integrate import H2IntegrateModel
from h2integrate.core.dict_utils import update_defaults
from h2integrate.core.file_utils import load_yaml, check_file_format_for_csv_generator
from pathlib import Path




# Load the configurations and run the model
config = load_yaml(Path(__file__).parent/"20_solar_electrolyzer_doe.yaml")

driver_config = load_yaml(Path(__file__).parent/config["driver_config"])
driver_config["general"]["folder_output"] = str(Path(__file__).parent/"ex_20_out")
driver_config["driver"]["parameter_sweep"]["filename"] = str(Path(__file__).parent/"csv_doe_cases.csv")
config["driver_config"] = driver_config
config["technology_config"] = Path(__file__).parent/"tech_config.yaml"
config["plant_config"] = Path(__file__).parent/"plant_config.yaml"
# csv_config_fn = driver_config["driver"]["parameter_sweep"]["filename"]

# try:
#     model = H2IntegrateModel(config)
#     model.run()
# except UserWarning as e:
#     print(f"Caught UserWarning: {e}")

# """
# To fix the issue with the UserWarning, we'll take the following steps to try and fix
# the bug in our CSV file:
# 1. Run the `check_file_format_for_csv_generator` method mentioned in the UserWarning
#   and create a new csv file that is hopefully free of errors
# 2. Make a new driver config file that has "filename" point to the new csv file created
#   in Step 1.
# 3. Make a new top-level config file that points to the updated driver config file
#   created in Step 2.
# """

# # Step 1
# new_csv_filename = check_file_format_for_csv_generator(
#     csv_config_fn,
#     driver_config,
#     check_only=False,
#     overwrite_file=True,
# )

# # Step 2
# updated_driver = update_defaults(
#     driver_config["driver"],
#     "filename",
#     new_csv_filename.name,
# )

# driver_config["driver"].update(updated_driver)
# print(f"New parameter sweep driver CSV file: {new_csv_filename}")

# # # Step 3
# config["driver_config"] = driver_config

# # # Rerun the model
model = H2IntegrateModel(config)
model.run()
