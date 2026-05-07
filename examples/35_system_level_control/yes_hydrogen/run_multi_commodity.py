import os

from h2integrate import EXAMPLE_DIR
from h2integrate.core.h2integrate_model import H2IntegrateModel


os.chdir(EXAMPLE_DIR / "35_system_level_control" / "yes_hydrogen")

##################################
# Create an H2I model with a fixed electricity load demand
h2i = H2IntegrateModel("wind_ng_demand.yaml")

h2i.setup()

# Run the model
h2i.run()

lcoe = h2i.prob.get_val("finance_subgroup_electricity.LCOE", units="USD/(MW*h)")
lcoh = h2i.prob.get_val("finance_subgroup_hydrogen.LCOH", units="USD/kg")

print(f"LCOE: {lcoe}")
print(f"LCOH: {lcoh}")
# Post-process the results
# h2i.post_process()
