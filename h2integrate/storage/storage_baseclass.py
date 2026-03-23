import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig
from h2integrate.core.validators import gt_zero, range_val, range_val_or_none
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


@define(kw_only=True)
class StoragePerformanceBaseConfig(BaseConfig):
    # Below is used in StoragePerformance and StorageAutoSizing
    commodity: str = field(converter=str.strip)
    commodity_rate_units: str = field(converter=str.strip)
    commodity_amount_units: str = field(default=None)
    charge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    discharge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    round_trip_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))

    # Below is only used in StoragePerformance
    max_discharge_rate: float | None = field(default=None)

    # Below is used in StoragePerformance and PySAMBattery
    max_capacity: float = field(validator=gt_zero)
    max_charge_rate: float = field(validator=gt_zero)
    charge_equals_discharge: bool = field(default=True)
    init_soc_fraction: float = field(validator=range_val(0, 1))

    # Below is used in all of them
    min_soc_fraction: float = field(validator=range_val(0, 1))
    max_soc_fraction: float = field(validator=range_val(0, 1))
    demand_profile: int | float | list = field()

    """
    Configuration class for the SimpleGenericStorage model.

     Attributes:
        commodity (str): Name of the commodity being stored (e.g., "hydrogen").
        commodity_units (str): Rate units of the commodity (e.g., "kg/h" or "kW").
        max_charge_rate (float): Maximum rate at which the commodity can be charged (in units
            per time step, e.g., "kg/time step"). This rate does not include the charge_efficiency.
        commodity_amount_units (str | None, optional): Units of the commodity as an amount
            (i.e., kW*h or kg). If not provided, defaults to commodity_rate_units*h.
        charge_equals_discharge (bool, optional): If True, set the max_discharge_rate equal to the
            max_charge_rate. If False, specify the max_discharge_rate as a value different than
            the max_charge_rate. Defaults to True.
        max_discharge_rate (float | None, optional): Maximum rate at which the commodity can be
            discharged (in units per time step, e.g., "kg/time step"). This rate does not include
            the discharge_efficiency. Only required if `charge_equals_discharge` is False.

    """

    def __attrs_post_init__(self):
        if (self.round_trip_efficiency is not None) and (
            self.charge_efficiency is None and self.discharge_efficiency is None
        ):
            # Calculate charge and discharge efficiencies from round-trip efficiency
            self.charge_efficiency = np.sqrt(self.round_trip_efficiency)
            self.discharge_efficiency = np.sqrt(self.round_trip_efficiency)
            self.round_trip_efficiency = None
        if self.charge_efficiency is None or self.discharge_efficiency is None:
            raise ValueError(
                "Exactly one of the following sets of parameters must be set: (a) "
                "`round_trip_efficiency`, or (b) both `charge_efficiency` "
                "and `discharge_efficiency`."
            )

        # Below only for StoragePerformance

        # if self.charge_equals_discharge:
        #     if (
        #         self.max_discharge_rate is not None
        #         and self.max_discharge_rate != self.max_charge_rate
        #     ):
        #         msg = (
        #             "Max discharge rate does not equal charge rate but charge_equals_discharge"
        #             f"is True. Discharge rate is {self.max_discharge_rate} and charge rate "
        #             f"is {self.max_charge_rate}."
        #         )
        #         raise ValueError(msg)

        #     self.max_discharge_rate = self.max_charge_rate

        # if not self.charge_equals_discharge and self.max_discharge_rate is None:
        #     msg = (
        #         "max_discharge_rate is a required key when charge_equals_discharge is True."
        #         "Please set a value for the max_discharge_rate."
        #     )
        #     raise ValueError(msg)

        # Below NOT used in PySAM battery
        # if self.commodity_amount_units is None:
        #     self.commodity_amount_units = f"({self.commodity_rate_units})*h"


class StoragePerformanceBase(PerformanceModelBaseClass):
    """
    Simple generic storage model that acts as a pass-through component.

    Note: this storage performance model is intended to be used with the
    `DemandOpenLoopStorageController` controller and has not been tested
    with other controllers.

    """

    def setup(self):
        # Below should be done in models that inherit it
        # self.commodity = self.config.commodity
        # self.commodity_rate_units = self.config.commodity_rate_units
        # self.commodity_amount_units = self.config.commodity_amount_units

        # Initialize standard performance model outptus
        super().setup()

        # Input timeseries
        self.add_input(
            f"{self.commodity}_demand",
            val=self.config.demand_profile,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} demand profile",
        )

        self.add_input(
            f"{self.commodity}_in",
            val=0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} input",
        )

        # Input storage design parameters
        if "max_charge_rate" in self.config.as_dict():
            self.add_input(
                "max_charge_rate",
                val=self.config.max_charge_rate,
                units=self.commodity_rate_units,
                desc="Storage charge rate",
            )

        if "max_capacity" in self.config.as_dict():
            self.add_input(
                "storage_capacity",
                val=self.config.max_capacity,
                units=self.commodity_ammount_units,
                desc="Storage capacity",
            )

        if not self.config.as_dict().get("charge_equals_discharge", True):
            # add if max_discharge_rate in config
            self.add_input(
                "max_discharge_rate",
                val=self.config.max_discharge_rate,
                units=self.commodity_rate_units,
                desc="Storage discharge rate",
            )

        # Storage design outputs:
        default_storage_duration = 0.0
        if "max_charge_rate" in self.config.as_dict() and "max_capacity" in self.config.as_dict():
            default_storage_duration = (self.config.max_capacity / self.config.max_charge_rate,)

        self.add_output(
            "storage_duration",
            val=default_storage_duration,
            units=f"({self.commodity_amount_units})/({self.commodity_rate_units})",
            desc="Storage duration capacity",
        )

        # Storage performance outputs
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

        self.add_output(
            f"storage_{self.commodity}_charge",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} input to storage only",
        )

        self.add_output(
            f"storage_{self.commodity}_out",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} input and output from storage",
        )

        self.add_input(
            f"{self.commodity}_demand",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} demand",
        )

        self.add_output(
            f"unmet_{self.commodity}_demand_out",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"Unmet {self.commodity} demand",
        )

        self.add_output(
            f"unused_{self.commodity}_out",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc="Unused generated commodity",
        )

        # TODO: update for bugfix in PR 615
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
        else:
            # using open-loop controller
            self.add_input(
                f"{self.commodity}_set_point",
                val=0.0,
                shape=self.n_timesteps,
                units=self.commodity_rate_units,
            )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        charge_rate = self.get_charge_rate(inputs)
        discharge_rate = self.get_discharge_rate(charge_rate, inputs)
        storage_capacity = self.get_storage_capacity(inputs)
        self.current_soc = self.get_init_soc(inputs)
        self.run_storage(
            charge_rate, discharge_rate, storage_capacity, inputs, outputs, discrete_inputs
        )

    def run_storage(
        self, charge_rate, discharge_rate, storage_capacity, inputs, outputs, discrete_inputs
    ):
        if "pyomo_dispatch_solver" in discrete_inputs:
            dispatch = discrete_inputs["pyomo_dispatch_solver"]
            # kwargs are tech-specific inputs to the simulate() method
            kwargs = {
                "charge_rate": charge_rate,
                "discharge_rate": discharge_rate,
                "storage_capacity": storage_capacity,
            }
            storage_commodity_out, soc = dispatch(self.simulate, kwargs, inputs)

        else:
            storage_commodity_out, soc = self.simulate(
                storage_dispatch_commands=inputs[f"{self.commodity}_set_point"],
                charge_rate=charge_rate,
                discharge_rate=discharge_rate,
                storage_capacity=storage_capacity,
            )

        # determine storage charge and discharge
        # storage_commodity_out is positive when the storage is discharged
        # and negative when the storage is charged
        storage_commodity_out = np.array(storage_commodity_out)

        # calculate combined commodity out from inflow source and storage
        # (note: storage_commodity_out is negative when charging)
        combined_commodity_out = inputs[f"{self.commodity}_in"] + storage_commodity_out

        # find the total commodity out to meet demand
        total_commodity_out = np.minimum(inputs[f"{self.commodity}_demand"], combined_commodity_out)

        # determine how much of the inflow commodity was unused
        unused_commodity = np.maximum(
            0, combined_commodity_out - inputs[f"{self.commodity}_demand"]
        )

        # determine how much demand was not met
        unmet_demand = np.maximum(0, inputs[f"{self.commodity}_demand"] - combined_commodity_out)

        # Storage design outputs
        if discharge_rate > 0:
            outputs["storage_duration"] = storage_capacity / discharge_rate
        else:
            outputs["storage_duration"] = 0

        # Storage specific timeseries outputs
        outputs[f"storage_{self.commodity}_charge"] = np.where(
            storage_commodity_out < 0, storage_commodity_out, 0
        )
        outputs[f"storage_{self.commodity}_discharge"] = np.where(
            storage_commodity_out > 0, storage_commodity_out, 0
        )

        outputs[f"unmet_{self.commodity}_demand_out"] = unmet_demand
        outputs[f"unused_{self.commodity}_out"] = unused_commodity
        outputs[f"storage_{self.commodity}_out"] = storage_commodity_out
        outputs[f"{self.commodity}_out"] = total_commodity_out

        outputs["SOC"] = soc

        # Performance model outputs
        outputs[f"rated_{self.commodity}_production"] = discharge_rate
        outputs[f"total_{self.commodity}_produced"] = np.sum(total_commodity_out)
        outputs[f"annual_{self.commodity}_produced"] = outputs[
            f"total_{self.commodity}_produced"
        ] * (1 / self.fraction_of_year_simulated)

        if outputs[f"rated_{self.commodity}_production"] <= 0:
            outputs["capacity_factor"] = 0.0
        else:
            outputs["capacity_factor"] = outputs[f"total_{self.commodity}_produced"] / (
                outputs[f"rated_{self.commodity}_production"] * self.n_timesteps
            )
        return outputs

    def simulate(
        self,
        storage_dispatch_commands: list,
        charge_rate: float,
        discharge_rate: float,
        storage_capacity: float,
        sim_start_index: int = 0,
    ):
        """Run the storage model over a control window of ``n_control_window`` timesteps.

        Iterates through ``storage_dispatch_commands`` one timestep at a time.
        A negative command requests charging; a positive command requests
        discharging.  Each command is clipped to the most restrictive of three
        limits before it is applied:

        1. **SOC headroom** - the remaining capacity (charge) or remaining
           stored commodity (discharge), converted to a rate via
           ``storage_capacity / dt_hr``.
        2. **Hardware rate limit** - ``charge_rate`` or ``discharge_rate``,
           divided by the corresponding efficiency so the limit is expressed
           in pre-efficiency rate units.
        3. **Commanded magnitude** - the absolute value of the dispatch command
           itself (we never exceed what was asked for).

        After clipping, the result is scaled by the charge or discharge
        efficiency to obtain the actual commodity flow into or out of the
        storage, and the SOC is updated accordingly.

        This method is separated from ``compute()`` so the Pyomo dispatch
        controller can call it directly to evaluate candidate schedules.

        Args:
            storage_dispatch_commands (array_like[float]):
                Dispatch set-points for each timestep in ``commodity_rate_units``.
                Negative values command charging; positive values command
                discharging.  Length must equal ``config.n_control_window``.
            charge_rate (float):
                Maximum commodity input rate to storage in
                ``commodity_rate_units`` (before charge efficiency is applied).
            discharge_rate (float):
                Maximum commodity output rate from storage in
                ``commodity_rate_units`` (before discharge efficiency is applied).
            storage_capacity (float):
                Rated storage capacity in ``commodity_amount_units``.
            sim_start_index (int, optional):
                Starting index for writing into persistent output arrays.
                Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray]
                storage_commodity_out_timesteps :
                    Commodity flow per timestep in ``commodity_rate_units``.
                    Positive = discharge (commodity leaving storage),
                    negative = charge (commodity entering storage).
                soc_timesteps :
                    State of charge at the end of each timestep, in percent
                    (0-100).
        """

        n = len(storage_dispatch_commands)
        storage_commodity_out_timesteps = np.zeros(n)
        soc_timesteps = np.zeros(n)

        # Early return when storage cannot operate: zero capacity or both
        # charge and discharge rates are zero.
        if storage_capacity <= 0 or (charge_rate <= 0 and discharge_rate <= 0):
            soc_timesteps[:] = self.current_soc * 100.0
            return storage_commodity_out_timesteps, soc_timesteps

        # Pre-compute scalar constants to avoid repeated attribute lookups
        # and redundant divisions inside the per-timestep loop.
        charge_eff = self.config.charge_efficiency
        discharge_eff = self.config.discharge_efficiency
        soc_max = self.config.max_soc_fraction
        soc_min = self.config.min_soc_fraction

        commands = np.asarray(storage_dispatch_commands, dtype=float)
        soc = float(self.current_soc)

        for t, cmd in enumerate(commands):
            if cmd < 0.0:
                # --- Charging ---
                # headroom: how much more commodity the storage can accept,
                # expressed as a rate (commodity_rate_units).
                headroom = (soc_max - soc) * storage_capacity / self.dt_hr

                # Clip to the most restrictive limit, then apply efficiency.
                # max(0, ...) guards against negative headroom when SOC
                # slightly exceeds soc_max.
                # correct headroom to not include charge_eff.
                actual_charge = max(0.0, min(headroom / charge_eff, charge_rate, -cmd)) * charge_eff

                # Update SOC (actual_charge is in post-efficiency units)
                soc += actual_charge / storage_capacity

                # Update the amount of commodity used to charge from the input stream
                # If charge_eff<1, more commodity is pulled from the input stream than
                # the commodity that goes into the storage.
                storage_commodity_out_timesteps[t] = -actual_charge / charge_eff
            else:
                # --- Discharging ---
                # headroom: how much commodity can still be drawn before
                # hitting the minimum SOC, expressed as a rate.
                headroom = (soc - soc_min) * storage_capacity / self.dt_hr

                # Clip to the most restrictive limit without applied efficiency.
                # Efficiency losses occur as energy leaves storage.
                actual_discharge = max(
                    0.0, min(headroom, discharge_rate / discharge_eff, cmd / discharge_eff)
                )

                # Update SOC (actual_discharge is before efficiency losses are applied.)
                soc -= actual_discharge / storage_capacity

                # If discharge_eff<1, then less commodity is output from the storage
                # than the commodity discharged from storage
                storage_commodity_out_timesteps[t] = actual_discharge * discharge_eff

            soc_timesteps[t] = soc * 100.0

        # Persist the final SOC so subsequent simulate() calls (e.g. from the
        # Pyomo controller across rolling windows) start where we left off.
        self.current_soc = soc
        return storage_commodity_out_timesteps, soc_timesteps


def dummy_function():
    # this function is required for initializing the pyomo control input and nothing else
    pass
