import os
from pathlib import Path

import pandas as pd

from h2integrate.core.dict_utils import update_defaults
from h2integrate.core.file_utils import check_file_format_for_csv_generator
from h2integrate.core.h2integrate_model import H2IntegrateModel
from h2integrate.core.inputs.validation import load_tech_yaml, load_plant_yaml, load_driver_yaml


os.chdir(Path(__file__).parent)

# First check the driver csv file
driver_config = load_driver_yaml("driver_config.yaml")
csv_config_fn = driver_config["driver"]["design_of_experiments"]["filename"]

# Step 1 - check file
new_csv_filename = check_file_format_for_csv_generator(
    csv_config_fn,
    driver_config,
    check_only=False,
    overwrite_file=False,
)
# Update driver config file with new csv file
updated_driver = update_defaults(
    driver_config["driver"],
    "filename",
    new_csv_filename,
)
driver_config["driver"].update(updated_driver)

# Load plant and tech configs
plant_config = load_plant_yaml("plant_config.yaml")
tech_config = load_tech_yaml("tech_config_h2a.yaml")

h2i_config = {
    "name": "H2Integrate_config",
    "system_summary": "SMR",
    "driver_config": driver_config,
    "technology_config": tech_config,
    "plant_config": plant_config,
}

h2i = H2IntegrateModel(h2i_config)

# Run the model
h2i.run()

# Post-process the results
h2i.post_process(summarize_sql=True)

csv_fpath = Path(str(h2i.recorder_path).replace(".sql", ".csv"))
df = pd.read_csv(csv_fpath)
df["finance_subgroup_hydrogen.LCOH (USD/kg)"].mean()
print(f"min LCOH is: {df['finance_subgroup_hydrogen.LCOH (USD/kg)'].min()}")
print(f"average LCOH is: {df['finance_subgroup_hydrogen.LCOH (USD/kg)'].mean()}")
print(f"max LCOH is: {df['finance_subgroup_hydrogen.LCOH (USD/kg)'].max()}")
