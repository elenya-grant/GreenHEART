from attrs import field, define
from hercules_controllers import ControllerHCST, ControllerOCGT

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


try:
    from hercules import HerculesOutput
    from hercules.hercules_model import HerculesModel
except ImportError:
    HerculesModel = None

try:
    import hycon.controllers as HyconControllers
    from hycon.interfaces import HerculesInterface
except ImportError:
    HerculesInterface = None


@define(kw_only=True)
class HerculesConfig(BaseConfig):
    temp: int = field(default=0.0)
    hercules_config: dict = field(default={})


class HerculesWrapper(PerformanceModelBaseClass):
    def initialize(self):
        super().initialize()
        self.commodity = "electricity"
        self.commodity_amount_units = "kW*h"
        self.commodity_rate_units = "kW"

        if HerculesModel is None:
            raise ImportError(
                "The `hercules` package is required to use the performance model. "
                "Install it via:\n"
                "pip install git+https://github.com/NatLabRockies/hercules.git"
            )
        if HerculesInterface is None:
            raise ImportError(
                "The `hycon` package is required to use the performance model. "
                "Install it via:\n"
                "pip install git+https://github.com/NatLabRockies/hycon.git"
            )

    def setup(self):
        super().setup()

        self.config = HerculesConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )

        self.add_input(
            "interconnect_limit",
            self.config.hercules_config["plant"]["interconnect_limit"],
            shape=1,
            units="kW",
        )
        for key, herc_tech_cnfg in self.config.hercules_config.items():
            if "component_type" not in herc_tech_cnfg:
                continue
            if (
                herc_tech_cnfg["component_type"] == "BatterySimple"
                or herc_tech_cnfg["component_type"] == "BatteryLithiumIon"
            ):
                self.battery_tech_name = key
                self.add_input(
                    "max_charge_rate",
                    val=herc_tech_cnfg["charge_rate"],
                    units="kW",
                    desc="Battery charge rate",
                )

                self.add_input(
                    "max_discharge_rate",
                    val=herc_tech_cnfg["discharge_rate"],
                    units="kW",
                    desc="Battery discharge rate",
                )

                self.add_input(
                    "storage_capacity",
                    val=herc_tech_cnfg["energy_capacity"],
                    units="kW*h",
                    desc="Battery storage capacity",
                )

            if herc_tech_cnfg["component_type"] == "OpenCycleGasTurbine":
                self.gas_tech_name = key
                self.add_input(
                    "natural_gas_system_capacity",
                    val=herc_tech_cnfg["rated_capacity"],
                    units="kW",
                    desc="Natural gas plant rated capacity in kW",
                )

            if herc_tech_cnfg["component_type"] == "HardCoalSteamTurbine":
                self.coal_steam_tech_name = key
                self.add_input(
                    "coal_steam_turbine_capacity",
                    val=herc_tech_cnfg["rated_capacity"],
                    units="kW",
                    desc="Hard coal steam turbine rated capacity",
                )

            if herc_tech_cnfg["component_type"] == "SolarPySAMPVWatts":
                self.pv_tech_name = key
                self.add_input(
                    "pv_capacity_DC",
                    val=herc_tech_cnfg["system_capacity"],
                    units="kW",
                    desc="PV rated capacity in DC",
                )
                self.add_output(
                    "pv_capacity_AC", val=0.0, units="kW", desc="PV rated capacity in AC"
                )

            if (
                herc_tech_cnfg["component_type"] == "WindFarm"
                or herc_tech_cnfg["component_type"] == "WindFarmSCADAPower"
            ):
                # not yet supported as a design variable
                self.includes_wind = True
                self.wind_tech_name = key
                pass

            if herc_tech_cnfg["component_type"] == "ElectrolyzerPlant":
                # not yet supported for a design variable
                pass

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # NOTE: need to add a controller, which is a class in hercules

        total_plant_capacity = 0
        output_columns = []
        # Update the Hercules dictionary
        self.config.hercules_config["plant"]["interconnect_limit"] = inputs["interconnect_limit"][0]
        if "pv_capacity_DC" in inputs:
            self.config.hercules_config[self.pv_tech_name]["system_capacity"] = inputs[
                "pv_capacity_DC"
            ][0]
            output_columns.append(f"{self.pv_tech_name}.power")
            total_plant_capacity += inputs["pv_capacity_DC"][0]
        if "storage_capacity" in inputs:
            self.config.hercules_config[self.battery_tech_name]["energy_capacity"] = inputs[
                "storage_capacity"
            ][0]
            self.config.hercules_config[self.battery_tech_name]["charge_rate"] = inputs[
                "max_charge_rate"
            ][0]
            self.config.hercules_config[self.battery_tech_name]["discharge_rate"] = inputs[
                "max_discharge_rate"
            ][0]
            output_columns.append(f"{self.battery_tech_name}.power")
            total_plant_capacity += inputs["max_discharge_rate"][0]

        if "natural_gas_system_capacity" in inputs:
            self.config.hercules_config[self.gas_tech_name]["rated_capacity"] = inputs[
                "natural_gas_system_capacity"
            ][0]
            output_columns.append(f"{self.gas_tech_name}.power")
            total_plant_capacity += inputs["natural_gas_system_capacity"][0]

        if "coal_steam_turbine_capacity" in inputs:
            self.config.hercules_config[self.coal_steam_tech_name]["rated_capacity"] = inputs[
                "coal_steam_turbine_capacity"
            ][0]
            output_columns.append(f"{self.coal_steam_tech_name}.power")
            total_plant_capacity += inputs["coal_steam_turbine_capacity"][0]

        # Initialize the Hercules model
        hmodel = HerculesModel(self.config.hercules_config)

        # Establish controllers based on options
        interface = HerculesInterface(hmodel.h_dict)
        wind_controller = None
        battery_controller = None
        pv_controller = None

        if "pv_capacity_DC" in inputs:
            pv_controller = HyconControllers.SolarPassthroughController(interface, hmodel.h_dict)

        if "storage_capacity" in inputs:
            battery_controller = HyconControllers.BatteryPassthroughController(
                interface, hmodel.h_dict
            )

        if "natural_gas_system_capacity" in inputs:
            # TODO: make PR to hercules with the controller
            ng_controller = ControllerOCGT(hmodel.h_dict, component_name=self.gas_tech_name)
            hmodel.assign_controller(ng_controller)
            pass
        if "coal_steam_turbine_capacity" in inputs:
            # TODO: make PR to hercules with the controller
            coal_controller = ControllerHCST(
                hmodel.h_dict, component_name=self.coal_steam_tech_name
            )
            hmodel.assign_controller(coal_controller)
            pass

        if self.includes_wind:
            output_columns.append(f"{self.wind_tech_name}.power")
            wind_controller = HyconControllers.WindFarmPowerTrackingController(
                interface, hmodel.h_dict
            )

        controller = HyconControllers.HybridSupervisoryControllerBase(
            interface,
            hmodel.h_dict,
            wind_controller=wind_controller,
            pv_controller=pv_controller,
            battery_controller=battery_controller,
        )

        # Assign the controller
        hmodel.assign_controller(controller)

        # Run the simulation
        hmodel.run()

        # Load outputs
        ho = HerculesOutput(hmodel.output_file)
        ho.df["plant.locally_generated_power"]
        ho.df["plant.power"]
        ho.df["time"]  # time in seconds
