import os

from h2integrate import EXAMPLE_DIR
from h2integrate.core.h2integrate_model import H2IntegrateModel
from h2integrate.core.inputs.validation import load_tech_yaml, load_plant_yaml, load_driver_yaml


this_dir = EXAMPLE_DIR / "xx_slc_draft"
os.chdir(this_dir)

plant_config = load_plant_yaml(this_dir / "plant_config.yaml")
driver_config = load_driver_yaml(this_dir / "driver_config.yaml")
tech_config = load_tech_yaml(this_dir / "tech_config.yaml")


top_level_config = {
    "plant_config": plant_config,
    "technology_config": tech_config,
    "driver_config": driver_config,
}
# Create an H2I model
h2i = H2IntegrateModel(top_level_config)

# Run the model
h2i.run()

# Post-process the results
h2i.post_process()
