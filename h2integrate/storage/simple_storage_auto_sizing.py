import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import range_val, range_val_or_none
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


@define(kw_only=True)
class StorageSizingModelConfig(BaseConfig):
    """Configuration class for the StorageAutoSizingModel.

    Fields include `commodity`, `commodity_rate_units`, and `demand_profile`.
    """

    commodity: str = field(converter=(str.strip, str.lower))
    commodity_rate_units: str = field(converter=str.strip)

    min_charge_fraction: float = field(validator=range_val(0, 1))
    max_charge_fraction: float = field(validator=range_val(0, 1))
    # init_charge_fraction: float = field(validator=range_val(0, 1)) #not used in sizing calc yet

    demand_profile: int | float | list = field(default=0.0)
    set_demand_as_avg_commodity_in: bool = field(default=True)

    charge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    discharge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    round_trip_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))

    commodity_amount_units: str = field(default=None)

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
                    "Exactly one of the following sets of parameters must be set: (a) "
                    "`round_trip_efficiency`, or (b) both `charge_efficiency` "
                    "and `discharge_efficiency`."
                )

            # Calculate charge and discharge efficiencies from round-trip efficiency
            self.charge_efficiency = np.sqrt(self.round_trip_efficiency)
            self.discharge_efficiency = np.sqrt(self.round_trip_efficiency)
        elif self.charge_efficiency is not None and self.discharge_efficiency is not None:
            # Ensure both charge and discharge efficiencies are provided
            pass
        else:
            raise ValueError(
                "Exactly one of the following sets of parameters must be set: (a) "
                "`round_trip_efficiency`, or (b) both `charge_efficiency` "
                "and `discharge_efficiency`."
            )

        # Set the default commodity_amount_units as the commodity_rate_units*h
        if self.commodity_amount_units is None:
            self.commodity_amount_units = f"({self.commodity_rate_units})*h"


class StorageAutoSizingModel(PerformanceModelBaseClass):
    """Performance model that calculates the storage charge rate and capacity needed
    to either:

    1. supply the commodity at a constant rate based on the commodity production profile or
    2. try to meet the commodity demand with the given commodity production profile.

    Then simulates performance of a basic storage component using the charge rate and
    capacity calculated.

    Note: this storage performance model is intended to be used with the
    `PassThroughOpenLoopController` controller and is not compatible with the
    `DemandOpenLoopStorageController` controller.

    Inputs:
        {commodity}_in (float): Input commodity flow timeseries (e.g., hydrogen production)
            used to estimate the demand if `commodity_demand_profile` is zero.
            - Units: Defined in `commodity_rate_units` (e.g., "kg/h").
        {commodity}_set_point (float): Input commodity flow timeseries (e.g., hydrogen production)
            used as the available input commodity to meet the demand.
        {commodity}_demand_profile (float): Demand profile of commodity.
            - Units: Defined in `commodity_rate_units` (e.g., "kg/h").

    Outputs:
        max_capacity (float): Maximum storage capacity of the commodity.
            - Units: in non-rate units, e.g., "kg" if `commodity_rate_units` is "kg/h"
        max_charge_rate (float): Maximum rate at which the commodity can be charged
            - Units: Defined in `commodity_rate_units` (e.g., "kg/h").
            Assumed to also be the discharge rate.
        {commodity}_out (np.ndarray): the commodity used to meet demand from the available
            input commodity and storage component. Defined in `commodity_rate_units`.
        total_{commodity}_produced (float): sum of commodity discharged from storage over
            the simulation. Defined in `commodity_amount_units`
        rated_{commodity}_production (float): maximum commodity that could be discharged
            in a timestep. Defined in `commodity_rate_units`
        annual_{commodity}_produced (np.ndarray): total commodity discharged per year.
            Defined in `commodity_amount_units/year`
        capacity_factor (np.ndarray): ratio of commodity discharged to the maximum
            commodity that could be discharged over the simulation.
            Defined as a ratio (units of `unitless`)

    """

    def setup(self):
        self.config = StorageSizingModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )

        self.commodity = self.config.commodity
        self.commodity_rate_units = self.config.commodity_rate_units
        self.commodity_amount_units = self.config.commodity_amount_units

        super().setup()

        # Inputs
        self.add_input(
            f"{self.commodity}_demand",
            val=self.config.demand_profile,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} demand profile timeseries",
        )

        self.add_input(
            f"{self.commodity}_in",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} input timeseries from production to storage",
        )

        self.add_input(
            f"{self.commodity}_set_point",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"{self.commodity} input set point from controller",
        )

        # Capacity outputs
        self.add_output(
            "storage_capacity",
            val=0.0,
            shape=1,
            units=self.commodity_amount_units,
        )

        self.add_output(
            "max_charge_rate",
            val=0.0,
            shape=1,
            units=self.commodity_rate_units,
        )
        self.add_output(
            "storage_duration",
            units=f"({self.commodity_amount_units})/({self.commodity_rate_units})",
            desc="Estimated storage duration based on max capacity and discharge rate",
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

        # Combined storage and input outputs
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

        self.dt_hr = self.dt / 3600  # convert from seconds to hours

    def compute(self, inputs, outputs):
        # Step 1: Auto-size the storage to meet the demand

        # Auto-size the fill rate as the max of the input commodity
        storage_max_fill_rate = (
            np.max(inputs[f"{self.commodity}_in"]) / self.config.charge_efficiency
        )

        # Set the demand profile
        if self.config.set_demand_as_avg_commodity_in:
            commodity_demand = np.mean(inputs[f"{self.commodity}_in"]) * np.ones(self.n_timesteps)
        else:
            commodity_demand = inputs[f"{self.commodity}_demand"]

        # NOTE: commodity_storage_soc is just an absolute value and is not a percentage.
        # Ideally would calculate as shortfall in future.
        # Size the storage capacity to meet the demand as much as possible
        commodity_storage_soc = np.cumsum(inputs[f"{self.commodity}_in"] - commodity_demand)
        minimum_soc = np.min(commodity_storage_soc)

        # Adjust soc so it's not negative.
        if minimum_soc < 0:
            commodity_storage_soc = [x + np.abs(minimum_soc) for x in commodity_storage_soc]

        # Calculate the maximum usable hydrogen storage capacity needed to meet the demand
        commodity_storage_capacity_kg = np.max(commodity_storage_soc) - np.min(
            commodity_storage_soc
        )

        # Calculate the storage capacity to account for SOC limits
        rated_storage_capacity = commodity_storage_capacity_kg / (
            self.config.max_charge_fraction - self.config.min_charge_fraction
        )

        # Estimate the initial SOC
        soc_amount = np.cumsum(inputs[f"{self.commodity}_in"] - commodity_demand)
        if np.min(soc_amount) < 0:
            soc_amount = soc_amount + np.abs(minimum_soc)
            # Adjust soc so it's never below the minimum
            min_soc_fraction = np.min(soc_amount) / rated_storage_capacity
            if min_soc_fraction < self.config.min_charge_fraction:
                soc_adjustment = (
                    self.config.min_charge_fraction - min_soc_fraction
                ) * rated_storage_capacity
                soc_amount = soc_amount + soc_adjustment
        self.current_soc = soc_amount[0] / rated_storage_capacity

        # Step 2: Simulate the storage performance based on the sizes calculated
        storage_commodity_out, soc = self.simulate(
            inputs[f"{self.commodity}_set_point"],
            storage_max_fill_rate,
            storage_max_fill_rate,
            rated_storage_capacity,
        )
        storage_commodity_out = np.array(storage_commodity_out)

        # calculate combined commodity out from inflow source and storage
        # (note: storage_commodity_out is negative when charging)
        combined_commodity_out = inputs[f"{self.commodity}_in"] + storage_commodity_out

        # find the total commodity out to meet demand
        total_commodity_out = np.minimum(commodity_demand, combined_commodity_out)

        # determine how much of the inflow commodity was unused
        unused_commodity = np.maximum(0, combined_commodity_out - commodity_demand)

        # determine how much demand was not met
        unmet_demand = np.maximum(0, commodity_demand - combined_commodity_out)

        # Output the storage performance outputs
        outputs[f"storage_{self.commodity}_charge"] = np.where(
            storage_commodity_out < 0, storage_commodity_out, 0
        )
        outputs[f"storage_{self.commodity}_discharge"] = np.where(
            storage_commodity_out > 0, storage_commodity_out, 0
        )
        outputs[f"unmet_{self.commodity}_demand_out"] = unmet_demand
        outputs[f"unused_{self.commodity}_out"] = unused_commodity
        outputs[f"storage_{self.commodity}_out"] = storage_commodity_out
        outputs["SOC"] = soc

        # commodity_out is the commodity_in - charge_storage + discharge_storage
        outputs[f"{self.commodity}_out"] = total_commodity_out

        # Output the calculated storage sizes (charge rate and capacity)
        outputs["max_charge_rate"] = storage_max_fill_rate
        outputs["storage_capacity"] = rated_storage_capacity
        outputs["storage_duration"] = outputs["storage_capacity"] / outputs["max_charge_rate"]

        # The rated_commodity_production is based on the discharge rate
        # (which is assumed equal to the charge rate)
        outputs[f"rated_{self.commodity}_production"] = storage_max_fill_rate

        # The total_commodity_produced is the sum of the commodity discharged from storage
        outputs[f"total_{self.commodity}_produced"] = np.sum(total_commodity_out)

        # Adjust the total_commodity_produced to a year-long simulation
        outputs[f"annual_{self.commodity}_produced"] = outputs[
            f"total_{self.commodity}_produced"
        ] * (1 / self.fraction_of_year_simulated)

        # Capacity factor is total discharged commodity / maximum discharged commodity possible
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
        soc_max = self.config.max_charge_fraction
        soc_min = self.config.min_charge_fraction

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
                actual_charge = max(0.0, min(headroom / charge_eff, charge_rate, -cmd)) * charge_eff

                # Update SOC (actual_charge is in post-efficiency units)
                soc += actual_charge / storage_capacity
                storage_commodity_out_timesteps[t] = -actual_charge / charge_eff
            else:
                # --- Discharging ---
                # headroom: how much commodity can still be drawn before
                # hitting the minimum SOC, expressed as a rate.
                headroom = (soc - soc_min) * storage_capacity / self.dt_hr

                # Clip and apply discharge efficiency.
                actual_discharge = max(
                    0.0, min(headroom, discharge_rate / discharge_eff, cmd / discharge_eff)
                )

                # Update SOC (actual_discharge is in post-efficiency units)
                soc -= actual_discharge / storage_capacity
                storage_commodity_out_timesteps[t] = actual_discharge * discharge_eff

            soc_timesteps[t] = soc * 100.0

        # Persist the final SOC so subsequent simulate() calls (e.g. from the
        # Pyomo controller across rolling windows) start where we left off.
        self.current_soc = soc
        return storage_commodity_out_timesteps, soc_timesteps
