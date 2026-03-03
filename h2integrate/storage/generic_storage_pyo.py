import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gt_zero, range_val, range_val_or_none
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


@define(kw_only=True)
class StoragePerformanceModelConfig(BaseConfig):
    """Configuration class for storage performance models.

    This class defines configuration parameters for simulating storage
    performance in PySAM system models. It includes
    specifications such as capacity, chemistry, state-of-charge limits,
    and reference module characteristics.

    Attributes:
        max_capacity (float):
            Maximum storage energy capacity in kilowatt-hours (kWh).
            Must be greater than zero.
        max_charge_rate (float):
            Rated power capacity of the storage in kilowatts (kW).
            Must be greater than zero.
        system_model_source (str):
            Source software for the system model. "hopp" source has not been brought
            over from HOPP yet. Options are:
                - "pysam"
        chemistry (str):
            storage chemistry option. "LDES" has not been brought over from HOPP yet.
            Supported values include:
                - PySAM: "LFPGraphite", "LMOLTO", "LeadAcid", "NMCGraphite"
        min_charge_percent (float):
            Minimum allowable state of charge as a fraction (0 to 1).
        max_charge_percent (float):
            Maximum allowable state of charge as a fraction (0 to 1).
        init_charge_percent (float):
            Initial state of charge as a fraction (0 to 1).
        n_control_window (int, optional):
            Number of timesteps in the control window. Defaults to 24.
        n_horizon_window (int, optional):
            Number of timesteps in the horizon window. Defaults to 48.
        control_variable (str):
            Control mode for the PySAM storage, either ``"input_power"``
            or ``"input_current"``.
        ref_module_capacity (int | float, optional):
            Reference module capacity in kilowatt-hours (kWh).
            Defaults to 400.
        ref_module_surface_area (int | float, optional):
            Reference module surface area in square meters (m²).
            Defaults to 30.
    """

    commodity: str = field()
    commodity_rate_units: str = field()

    max_capacity: float = field(validator=gt_zero)
    max_charge_rate: float = field(validator=gt_zero)

    min_charge_percent: float = field(validator=range_val(0, 1))
    max_charge_percent: float = field(validator=range_val(0, 1))
    init_charge_percent: float = field(validator=range_val(0, 1))
    n_control_window: int = field(validator=gt_zero, default=24)

    commodity_amount_units: str = field(default=None)
    max_discharge_rate: float | None = field(default=None)
    charge_equals_discharge: bool = field(default=True)

    charge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    discharge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    round_trip_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))

    # n_horizon_window: int = field(validator=gt_zero, default=48)
    # control_variable: str = field(
    #     default="input_power", validator=contains(["input_power", "input_current"])
    # )
    def __attrs_post_init__(self):
        """
        Post-initialization logic to validate and calculate efficiencies.

        Ensures that either `charge_efficiency` and `discharge_efficiency` are provided,
        or `round_trip_efficiency` is provided. If `round_trip_efficiency` is provided,
        it calculates `charge_efficiency` and `discharge_efficiency` as the square root
        of `round_trip_efficiency`.
        """
        if self.round_trip_efficiency is not None:
            if self.charge_efficiency is not None or self.discharge_efficiency is not None:
                raise ValueError(
                    "Provide either `round_trip_efficiency` or both `charge_efficiency` "
                    "and `discharge_efficiency`, but not both."
                )
            # Calculate charge and discharge efficiencies from round-trip efficiency
            self.charge_efficiency = np.sqrt(self.round_trip_efficiency)
            self.discharge_efficiency = np.sqrt(self.round_trip_efficiency)
        elif self.charge_efficiency is not None and self.discharge_efficiency is not None:
            # Ensure both charge and discharge efficiencies are provided
            pass
        else:
            raise ValueError(
                "You must provide either `round_trip_efficiency` or both "
                "`charge_efficiency` and `discharge_efficiency`."
            )

        if self.charge_equals_discharge:
            if (
                self.max_discharge_rate is not None
                and self.max_discharge_rate != self.max_charge_rate
            ):
                msg = (
                    "Max discharge rate does not equal max charge rate but charge_equals_discharge "
                    f"is True. Discharge rate is {self.max_discharge_rate} and charge rate "
                    f"is {self.max_charge_rate}."
                )
                raise ValueError(msg)

            self.max_discharge_rate = self.max_charge_rate

        if self.commodity_amount_units is None:
            self.commodity_amount_units = f"({self.commodity_rate_units})*h"


class StoragePerformanceModel(PerformanceModelBaseClass):
    """OpenMDAO component wrapping the PySAM storage Performance model.

    Attributes:
        config (PySAMBatteryPerformanceModelConfig):
            Configuration parameters for the storage performance model.
        system_model (BatteryStateful):
            Instance of the PySAM BatteryStateful model, initialized with
            the selected chemistry and configuration parameters.
        outputs (BatteryOutputs):
            Container for simulation outputs such as SOC, chargeable/dischargeable
            power, unmet demand, and unused commodities.
        unmet_demand (float):
            Tracks unmet demand during simulation (kW).
        unused_commodity (float):
            Tracks unused commodity during simulation (kW).

    Inputs:
        max_charge_rate (float):
            storage charge rate in kilowatts per hour (kW).
        storage_capacity (float):
            Total energy storage capacity in kilowatt-hours (kWh).
        electricity_demand (ndarray):
            Power demand time series (kW).
        electricity_in (ndarray):
            Commanded input electricity (kW), typically from dispatch.

    Outputs:
        unmet_demand_out (ndarray):
            Remaining unmet demand after discharge (kW).
        unused_commodity_out (ndarray):
            Unused energy not absorbed by the storage (kW).
        electricity_out (ndarray):
            Dispatched electricity to meet demand (kW), including electricity from
            electricity_in that was never used to charge the storage and
            storage_electricity_discharge.
        SOC (ndarray):
            storage state of charge (%).
        storage_electricity_discharge (ndarray):
            Electricity output from the storage model (kW).

    Methods:
        setup():
            Defines model inputs, outputs, configuration, and connections
            to plant-level dispatch (if applicable).
        compute(inputs, outputs, discrete_inputs, discrete_outputs):
            Runs the PySAM BatteryStateful model for a simulation timestep,
            updating outputs such as SOC, charge/discharge limits, unmet
            demand, and unused commodities.
        simulate(electricity_in, electricity_demand, time_step_duration, control_variable,
            sim_start_index=0):
            Simulates the storage behavior across timesteps using either
            input power or input current as control. This method is similar to what is
            provided in typical compute methods in H2Integrate for running models, but
            needs to be a separate method here to allow the dispatch function to call
            and manage the performance model.
        _set_control_mode(control_mode=1.0, input_power=0.0, input_current=0.0,
            control_variable="input_power"):
            Sets the storage control mode (power or current).

    Notes:
        - Default timestep is 1 hour (``dt=1.0``).
        - State of charge (SOC) bounds are set using the configuration's
          ``min_charge_percent`` and ``max_charge_percent``.
        - If a Pyomo dispatch solver is provided, the storage will simulate
          dispatch decisions using solver inputs.
    """

    def setup(self):
        """Set up the PySAM storage Performance model in OpenMDAO.

        Initializes the configuration, defines inputs/outputs for OpenMDAO,
        and creates a `BatteryStateful` instance with the selected chemistry.
        If dispatch connections are specified, it also sets up a discrete
        input for Pyomo solver integration.
        """
        self.config = StoragePerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )

        self.commodity = self.config.commodity
        self.commodity_rate_units = self.config.commodity_rate_units
        self.commodity_amount_units = self.config.commodity_amount_units
        super().setup()

        self.add_input(
            f"{self.commodity}_in",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} input to storage",
        )

        self.add_output(
            "SOC",
            val=0.0,
            shape=self.n_timesteps,
            units="percent",
            desc="State of charge of storage",
        )

        self.add_output(
            f"storage_{self.commodity}_discharge",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} output from storage only",
        )

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=self.config.commodity_rate_units,
            desc="Storage charge rate",
        )

        self.add_input(
            "storage_capacity",
            val=self.config.max_capacity,
            units=self.commodity_amount_units,
            desc="Storage capacity",
        )

        self.add_input(
            f"{self.commodity}_demand",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc="Power demand",
        )

        self.add_output(
            f"unmet_{self.commodity}_demand_out",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc="Unmet power demand",
        )

        self.add_output(
            f"unused_{self.commodity}_out",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc="Unused generated commodity",
        )

        # Initialize the PySAM BatteryStateful model with defaults

        self.dt_hr = int(self.options["plant_config"]["plant"]["simulation"]["dt"]) / (
            60**2
        )  # convert from seconds to hours

        # create inputs for pyomo control model
        if "tech_to_dispatch_connections" in self.options["plant_config"]:
            # get technology group name
            # TODO: The split below seems brittle
            self.tech_group_name = self.pathname.split(".")
            for _source_tech, intended_dispatch_tech in self.options["plant_config"][
                "tech_to_dispatch_connections"
            ]:
                if any(intended_dispatch_tech in name for name in self.tech_group_name):
                    self.add_discrete_input("pyomo_dispatch_solver", val=dummy_function)
                    break

    def compute(self, inputs, outputs, discrete_inputs=[], discrete_outputs=[]):
        """Run the PySAM storage model for one simulation step.

        Configures the storage stateful model parameters (SOC limits, timestep,
        thermal properties, etc.), executes the simulation, and stores the
        results in OpenMDAO outputs.

        Args:
            inputs (dict):
                Continuous input values (e.g., electricity_in, electricity_demand).
            outputs (dict):
                Dictionary where model outputs (SOC, P_chargeable, unmet demand, etc.)
                are written.
            discrete_inputs (dict):
                Discrete inputs such as control mode or Pyomo solver.
            discrete_outputs (dict):
                Discrete outputs (unused in this component).
        """
        # Size the storage based on inputs -> method brought from HOPP
        if self.config.charge_equals_discharge:
            max_discharge_rate = inputs["max_charge_rate"].item()
        else:
            max_discharge_rate = float(self.config.max_discharge_rate)

        self.current_soc = self.config.init_charge_percent
        self.unmet_demand = 0.0
        self.unused_commodity = 0.0

        if "pyomo_dispatch_solver" in discrete_inputs:
            # Simulate the storage with provided dispatch inputs
            dispatch = discrete_inputs["pyomo_dispatch_solver"]
            # kwargs are tech-specific inputs to the simulate() method
            kwargs = {
                "charge_rate": inputs["max_charge_rate"][0],
                "discharge_rate": max_discharge_rate,
                "storage_capacity": inputs["storage_capacity"][0],
            }
            (
                total_commodity_out,
                storage_commodity_out,
                unmet_demand,
                unused_commodity,
                soc,
            ) = dispatch(self.simulate, kwargs, inputs)

        else:
            # Simulate the storage with provided inputs and no controller.
            # This essentially asks for discharge when demand exceeds input
            # and requests charge when input exceeds demand

            # estimate required dispatch commands
            pseudo_commands = inputs[f"{self.commodity}_demand"] - inputs[f"{self.commodity}_in"]

            storage_power, soc = self.simulate(
                storage_dispatch_commands=pseudo_commands,
                charge_rate=inputs["charge_rate"][0],
                discharge_rate=max_discharge_rate,
                storage_capacity=inputs["storage_capacity"][0],
            )

            # determine storage discharge
            storage_commodity_out = [np.max([0, storage_power[i]]) for i in range(self.n_timesteps)]

            # calculate combined power out from inflow source and storage (note: storage_power is
            # negative when charging)
            combined_power_out = inputs[f"{self.commodity}_in"] + storage_power

            # find the total power out to meet demand
            np.minimum(inputs[f"{self.commodity}_demand"], combined_power_out)

            # determine how much of the inflow electricity was unused
            unused_commodity = [
                np.max([0, combined_power_out[i] - inputs[f"{self.commodity}_demand"][i]])
                for i in range(self.n_timesteps)
            ]

            # determine how much demand was not met
            unmet_demand = [
                np.max([0, inputs[f"{self.commodity}_demand"][i] - combined_power_out[i]])
                for i in range(self.n_timesteps)
            ]

        outputs[f"unmet_{self.commodity}_demand_out"] = unmet_demand
        outputs[f"unused_{self.commodity}_out"] = unused_commodity
        outputs[f"storage_{self.commodity}_discharge"] = storage_commodity_out
        outputs[f"{self.commodity}_out"] = total_commodity_out
        outputs["SOC"] = soc
        outputs[f"rated_{self.commodity}_production"] = max_discharge_rate

        outputs[f"total_{self.commodity}_produced"] = np.sum(total_commodity_out)
        outputs[f"annual_{self.commodity}_produced"] = outputs[
            f"total_{self.commodity}_produced"
        ] * (1 / self.fraction_of_year_simulated)
        outputs["capacity_factor"] = outputs[f"total_{self.commodity}_produced"] / (
            outputs[f"rated_{self.commodity}_production"] * self.n_timesteps
        )

    def simulate(
        self,
        storage_dispatch_commands: list,
        charge_rate: float,
        discharge_rate: float,
        storage_capacity: float,
        sim_start_index: int = 0,
    ):
        """Run the PySAM BatteryStateful model over a control window.

        Applies a sequence of dispatch commands (positive = discharge, negative = charge)
        one timestep at a time. Each command is clipped to allowable instantaneous
        charge / discharge limits derived from:
          1. Rated power (config.max_charge_rate)
          2. PySAM internal estimates (P_chargeable / P_dischargeable)
          3. Remaining energy headroom vs. SOC bounds

        The method updates internal rolling arrays in self.outputs in-place using
        sim_start_index as an offset (enabling sliding / receding horizon logic).

        The simulate method is much of what would normally be in the compute() method
        of a component, but is separated into its own function here to allow the dispatch()
        method to manage calls to the performance model.

        Args:
            storage_dispatch_commands : Sequence[float]
                Commanded power per timestep (kW). Negative = charge, positive = discharge.
                Length should be = config.n_control_window.
            time_step_duration : float | Sequence[float]
                Timestep duration in hours. Scalar applied uniformly or sequence matching
                len(storage_dispatch_commands).
            control_variable : str
                PySAM control input to set each step ("input_power" or "input_current").
            sim_start_index : int, optional
                Starting index for writing into persistent output arrays (default 0).

        Returns:
            tuple[np.ndarray, np.ndarray]
                (storage_power_kW, soc_percent)
                storage_power_kW : array of PySAM P values (kW) per timestep
                                    (positive = discharge, negative = charge).
                soc_percent      : array of SOC values (%) per timestep.

        Notes:
            - SOC bounds may still be exceeded slightly due to PySAM internal dynamics.
            - self.outputs.stateful_attributes are updated only if the attribute exists
            in StatePack or StateCell.
            - self.outputs.component_attributes (e.g., unmet_demand) are not modified here;
            they are populated in compute(), unless an external dispatcher manages them.
        """

        # Loop through the provided input power/current (decided by control_variable)

        # initialize outputs
        storage_power_out_timesteps = np.zeros(self.config.n_control_window)
        soc_timesteps = np.zeros(self.config.n_control_window)

        soc = float(self.current_soc)
        for t, dispatch_command_t in enumerate(storage_dispatch_commands):
            # get storage SOC at time t

            # if commanded to charge
            if dispatch_command_t < 0:
                # available charge is positive?
                available_charge = float(
                    (self.config.max_charge_percent - soc) * storage_capacity / self.dt_hr
                )
                max_chargeable = (
                    np.min(
                        [
                            available_charge,
                            charge_rate / self.config.charge_efficiency,
                            -1 * dispatch_command_t,
                        ]
                    )
                    * self.config.charge_efficiency
                )
                if dispatch_command_t < -max_chargeable:
                    dispatch_command_t = -max_chargeable

            else:
                # also positive
                available_discharge = float(
                    (soc - self.config.min_charge_percent) * storage_capacity / self.dt_hr
                )
                max_dischargeable = (
                    np.min(
                        [
                            available_discharge,
                            discharge_rate / self.config.discharge_efficiency,
                            dispatch_command_t,
                        ]
                    )
                    * self.config.discharge_efficiency
                )
                if dispatch_command_t > max_dischargeable:
                    dispatch_command_t = max_dischargeable

            # if storage soc is outside the set bounds, discharge storage down to set bounds
            if (
                soc > self.config.max_charge_percent
            ) and dispatch_command_t < 0:  # and (dispatch_command_t <= 0):
                dispatch_command_t = 0.0

            if dispatch_command_t < 0:
                # charge: increase soc and negative storage_power_out
                soc += max_chargeable / storage_capacity
                storage_power_out_timesteps[t] = -1 * max_chargeable
            else:
                # discharge: decrease soc and positive storage_power_out
                soc -= max_dischargeable / storage_capacity
                storage_power_out_timesteps[t] = max_dischargeable

            # save outputs
            soc_timesteps[t] = soc * 100

        self.current_soc = soc
        return storage_power_out_timesteps, soc_timesteps


def dummy_function():
    # this function is required for initializing the pyomo control input and nothing else
    pass
