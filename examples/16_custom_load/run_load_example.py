from h2integrate.core.h2integrate_model import H2IntegrateModel


# Create a GreenHEART model
gh = H2IntegrateModel("load_demand.yaml")

# Run the model
gh.run()

gh.post_process()

# gh.plant.get_val("electrical_load_demand.electricity_out")
# gh.plant.get_val("electrical_load_demand.electricity_curtailed")
gh.plant.get_val("natural_gas_plant.electricity_in", units="MW")
gh.plant.get_val("wind.electricity_out", units="MW")

# check that the below two lines equal each other
gh.plant.get_val("wind.electricity_out", units="MW") + gh.plant.get_val(
    "natural_gas_plant.electricity_in", units="MW"
)
gh.plant.get_val("electrical_load_demand.original_electricity_demand", units="MW")

# check that if NG plant is not constrained by capacity (?), that the below two lines are equal
gh.plant.get_val("natural_gas_plant.electricity_in", units="MW")  # electricity demand
gh.plant.get_val("electrical_load_demand.electricity_demand", units="MW")

# check that the two lines below are equal if not constrained by capacity
gh.plant.get_val("electrical_load_demand.electricity_supplied", units="MW")
gh.plant.get_val("wind.electricity_out", units="MW")

# check for curtailed power
gh.plant.get_val("electrical_load_demand.electricity_curtailed", units="MW")

# check that
natural_gas_outputs = ["natural_gas_consumed", "electricity_out"]
natural_gas_inputs = ["natural_gas_in", "electricity_in"]

# gh.create_technology_models()
# gh.create_financial_model()
# gh.connect_technologies()
# gh.create_driver_model()
