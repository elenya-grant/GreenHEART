from h2integrate.resource.river import RiverResource
from h2integrate.transporters.pipe import PipePerformanceModel
from h2integrate.transporters.cable import CablePerformanceModel
from h2integrate.converters.steel.steel import SteelPerformanceModel, SteelCostAndFinancialModel
from h2integrate.core.profast_financial import ProFastComp
from h2integrate.converters.wind.wind_plant import WindPlantCostModel, WindPlantPerformanceModel
from h2integrate.transporters.power_combiner import CombinerPerformanceModel
from h2integrate.converters.hopp.hopp_wrapper import HOPPComponent
from h2integrate.converters.solar.solar_pysam import PYSAMSolarPlantPerformanceModel
from h2integrate.storage.hydrogen.eco_storage import H2Storage
from h2integrate.converters.nitrogen.simple_ASU import SimpleASUCostModel, SimpleASUPerformanceModel
from h2integrate.storage.hydrogen.tank_baseclass import (
    HydrogenTankCostModel,
    HydrogenTankPerformanceModel,
)
from h2integrate.controllers.openloop_controllers import (
    DemandOpenLoopController,
    PassThroughOpenLoopController,
)
from h2integrate.converters.hydrogen.wombat_model import WOMBATElectrolyzerModel
from h2integrate.converters.wind.wind_plant_pysam import PYSAMWindPlantPerformanceModel
from h2integrate.converters.ammonia.ammonia_synloop import (
    AmmoniaSynLoopCostModel,
    AmmoniaSynLoopPerformanceModel,
)
from h2integrate.converters.water.desal.desalination import (
    ReverseOsmosisCostModel,
    ReverseOsmosisPerformanceModel,
)
from h2integrate.converters.hydrogen.basic_cost_model import BasicElectrolyzerCostModel
from h2integrate.converters.hydrogen.pem_electrolyzer import (
    ElectrolyzerCostModel,
    ElectrolyzerPerformanceModel,
)
from h2integrate.converters.solar.atb_res_com_pv_cost import ATBResComPVCostModel
from h2integrate.converters.solar.atb_utility_pv_cost import ATBUtilityPVCostModel
from h2integrate.converters.methanol.smr_methanol_plant import (
    SMRMethanolPlantCostModel,
    SMRMethanolPlantFinanceModel,
    SMRMethanolPlantPerformanceModel,
)
from h2integrate.converters.ammonia.simple_ammonia_model import (
    SimpleAmmoniaCostModel,
    SimpleAmmoniaPerformanceModel,
)
from h2integrate.converters.methanol.co2h_methanol_plant import (
    CO2HMethanolPlantCostModel,
    CO2HMethanolPlantFinanceModel,
    CO2HMethanolPlantPerformanceModel,
)
from h2integrate.converters.hydrogen.singlitico_cost_model import SingliticoCostModel
from h2integrate.converters.co2.marine.direct_ocean_capture import DOCCostModel, DOCPerformanceModel
from h2integrate.converters.hydrogen.eco_tools_pem_electrolyzer import (
    ECOElectrolyzerPerformanceModel,
)
from h2integrate.converters.water_power.hydro_plant_run_of_river import (
    RunOfRiverHydroCostModel,
    RunOfRiverHydroPerformanceModel,
)
from h2integrate.converters.hydrogen.geologic.natural_geoh2_plant import (
    NaturalGeoH2CostModel,
    NaturalGeoH2FinanceModel,
    NaturalGeoH2PerformanceModel,
)
from h2integrate.converters.co2.marine.ocean_alkalinity_enhancement import (
    OAECostModel,
    OAEPerformanceModel,
    OAECostAndFinancialModel,
)
from h2integrate.converters.hydrogen.geologic.stimulated_geoh2_plant import (
    StimulatedGeoH2CostModel,
    StimulatedGeoH2FinanceModel,
    StimulatedGeoH2PerformanceModel,
)


supported_models = {
    # Resources
    "river_resource": RiverResource,
    # Converters
    # Converters (Renewable):
    "hopp": HOPPComponent,
    # Converters (Renewable): Wind
    "wind_plant_performance": WindPlantPerformanceModel,
    "wind_plant_cost": WindPlantCostModel,
    "pysam_wind_plant_performance": PYSAMWindPlantPerformanceModel,
    # Converter (Renewable): Solar-PV
    "pysam_solar_plant_performance": PYSAMSolarPlantPerformanceModel,
    "atb_utility_pv_cost": ATBUtilityPVCostModel,
    "atb_comm_res_pv_cost": ATBResComPVCostModel,
    # Converter (Renewable): Hydropower
    "run_of_river_hydro_performance": RunOfRiverHydroPerformanceModel,
    "run_of_river_hydro_cost": RunOfRiverHydroCostModel,
    # Converter: Hydrogen Electrolysis
    "pem_electrolyzer_performance": ElectrolyzerPerformanceModel,
    "pem_electrolyzer_cost": ElectrolyzerCostModel,
    "eco_pem_electrolyzer_performance": ECOElectrolyzerPerformanceModel,
    "singlitico_electrolyzer_cost": SingliticoCostModel,
    "basic_electrolyzer_cost": BasicElectrolyzerCostModel,
    "wombat": WOMBATElectrolyzerModel,
    # Converter: Nitrogen
    "simple_ASU_cost": SimpleASUCostModel,
    "simple_ASU_performance": SimpleASUPerformanceModel,
    # Converter: Desal
    "reverse_osmosis_desalination_performance": ReverseOsmosisPerformanceModel,
    "reverse_osmosis_desalination_cost": ReverseOsmosisCostModel,
    # Converter: Ammonia
    "simple_ammonia_performance": SimpleAmmoniaPerformanceModel,
    "simple_ammonia_cost": SimpleAmmoniaCostModel,
    "synloop_ammonia_performance": AmmoniaSynLoopPerformanceModel,
    "synloop_ammonia_cost": AmmoniaSynLoopCostModel,
    # Converter: Steel
    "steel_performance": SteelPerformanceModel,
    "steel_cost": SteelCostAndFinancialModel,
    # Converter: Methanol
    "smr_methanol_plant_performance": SMRMethanolPlantPerformanceModel,
    "smr_methanol_plant_cost": SMRMethanolPlantCostModel,
    "smr_methanol_plant_financial": SMRMethanolPlantFinanceModel,
    "co2h_methanol_plant_performance": CO2HMethanolPlantPerformanceModel,
    "co2h_methanol_plant_cost": CO2HMethanolPlantCostModel,
    "co2h_methanol_plant_financial": CO2HMethanolPlantFinanceModel,
    # Converter: Carbon Capture
    "direct_ocean_capture_performance": DOCPerformanceModel,
    "direct_ocean_capture_cost": DOCCostModel,
    "ocean_alkalinity_enhancement_performance": OAEPerformanceModel,
    "ocean_alkalinity_enhancement_cost": OAECostModel,
    "ocean_alkalinity_enhancement_cost_financial": OAECostAndFinancialModel,
    # Converter: Geologic Hydrogen
    "natural_geoh2_performance": NaturalGeoH2PerformanceModel,
    "natural_geoh2_cost": NaturalGeoH2CostModel,
    "natural_geoh2_financial": NaturalGeoH2FinanceModel,
    "stimulated_geoh2_performance": StimulatedGeoH2PerformanceModel,
    "stimulated_geoh2_cost": StimulatedGeoH2CostModel,
    "stimulated_geoh2_financial": StimulatedGeoH2FinanceModel,
    # Transport
    "cable": CablePerformanceModel,
    "pipe": PipePerformanceModel,
    "combiner_performance": CombinerPerformanceModel,
    # Control
    "pass_through_controller": PassThroughOpenLoopController,
    "demand_open_loop_controller": DemandOpenLoopController,
    # Storage
    "h2_storage": H2Storage,
    "hydrogen_tank_performance": HydrogenTankPerformanceModel,
    "hydrogen_tank_cost": HydrogenTankCostModel,
    # Finance
    "ProFastComp": ProFastComp,
}

electricity_producing_techs = ["wind", "solar", "pv", "river", "hopp"]

commodity_to_common_units = {
    "hydrogen": "kg",
    "steel": "t",  # metric ton
    "methanol": "kg",  # double check
    "ammonia": "kg",
    "nitrogen": "kg",
    "carbon": "kg",
    "electricity": "kW",
    "water": "kg",
}

commodity_to_nickname = {
    "hydrogen": "H2",
    # "steel": "",
    # "methanol": "",
    "ammonia": "NH3",
    "nitrogen": "N2",
    "carbon": "CO2",
    # "electricity": "",
    "water": "H2O",
}

commodity_to_model_kwargs = {
    "hydrogen": ["electrolyzer", "geoh2", "h2_storage", "wombat", "hydrogen_tank"],
    "steel": ["steel"],
    "methanol": ["methanol", "smr"],
    "ammonia": ["ammonia"],
    "nitrogen": ["air_separator", "asu"],
    "carbon": ["doc", "direct_ocean_capture", "ocean_alkalinity_enhancement"],
    "electricity": ["wind", "solar", "pv", "river", "hydro", "hopp"],
    "water": ["desal", "desalination"],
}


# These handle their own financials
independent_financial_models = [
    "steel_cost",
    "stimulated_geoh2_financial",
    "natural_geoh2_financial",
    "smr_methanol_plant_financial",
    "co2h_methanol_plant_financial",
]


combined_performance_and_cost_models = ["hopp", "h2_storage", "wombat"]
