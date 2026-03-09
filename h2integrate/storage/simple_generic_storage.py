import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gte_zero, range_val, range_val_or_none
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


@define(kw_only=True)
class SimpleGenericStorageConfig(BaseConfig):
    commodity: str = field()
    commodity_rate_units: str = field()  # TODO: update to commodity_rate_units
    max_charge_rate: float = field(validator=gte_zero)
    max_capacity: float = field()
    max_charge_fraction: float = field(validator=range_val(0, 1))
    min_charge_fraction: float = field(validator=range_val(0, 1))
    init_charge_fraction: float = field(validator=range_val(0, 1))
    charge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    discharge_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    round_trip_efficiency: float | None = field(default=None, validator=range_val_or_none(0, 1))
    demand_profile: int | float | list = field()

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


class SimpleGenericStorage(PerformanceModelBaseClass):
    """
    Simple generic storage model that acts as a pass-through component.

    Note: this storage performance model is intended to be used with the
    `DemandOpenLoopStorageController` controller and has not been tested
    with other controllers.

    """

    def setup(self):
        self.config = SimpleGenericStorageConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )
        self.commodity = self.config.commodity
        self.commodity_rate_units = self.config.commodity_rate_units
        self.commodity_amount_units = f"({self.commodity_rate_units})*h"
        super().setup()
        self.add_input(
            f"{self.commodity}_set_point",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
        )
        self.add_input(
            f"{self.commodity}_in",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
            desc=f"Amount of {self.commodity} demand that has already been supplied",
        )

        self.add_input(
            f"{self.commodity}_demand",
            units=f"{self.config.commodity_rate_units}",
            val=self.config.demand_profile,
            shape=self.n_timesteps,
            desc=f"{self.commodity} demand profile timeseries",
        )
        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=self.config.commodity_rate_units,
            desc="Storage charge/discharge rate",
        )
        self.add_input(
            "storage_capacity",
            val=self.config.max_capacity,
            units=self.commodity_amount_units,
            desc="Storage capacity",
        )

        self.dt_hr = int(self.options["plant_config"]["plant"]["simulation"]["dt"]) / (
            60**2
        )  # convert from seconds to hours

    def compute(self, inputs, outputs):
        self.current_soc = float(self.config.init_charge_fraction)

        storage_commodity_out, soc = self.simulate(
            storage_dispatch_commands=inputs[f"{self.commodity}_set_point"],
            charge_rate=inputs["max_charge_rate"][0],
            discharge_rate=inputs["max_charge_rate"][0],
            storage_capacity=inputs["storage_capacity"][0],
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
        # unused_commodity = np.maximum(
        #     0, combined_commodity_out - inputs[f"{self.commodity}_demand"]
        # )

        # # determine how much demand was not met
        # unmet_demand = np.maximum(
        #     0, inputs[f"{self.commodity}_demand"] - combined_commodity_out
        # )

        discharge_storage = np.where(storage_commodity_out > 0, storage_commodity_out, 0)

        # TODO: update the below outputs
        # Pass the commodity_out as the commodity_set_point
        outputs[f"{self.commodity}_out"] = total_commodity_out

        # Set the rated commodity production from the max_charge_rate input
        outputs[f"rated_{self.commodity}_production"] = inputs["max_charge_rate"]

        # Calculate the total and annual commodity produced
        outputs[f"total_{self.commodity}_produced"] = discharge_storage.sum()
        outputs[f"annual_{self.commodity}_produced"] = outputs[
            f"total_{self.commodity}_produced"
        ] * (1 / self.fraction_of_year_simulated)

        # Calculate the maximum theoretical commodity production over the simulation
        rated_production = (
            outputs[f"rated_{self.commodity}_production"] * self.n_timesteps * (self.dt / 3600)
        )

        outputs["capacity_factor"] = outputs[f"total_{self.commodity}_produced"] / rated_production

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

        # max_charge_input / max_discharge_input are the hardware rate limits
        # expressed in *pre-efficiency* rate units so they can be compared
        # directly against the SOC headroom and the raw command magnitude.
        max_charge_input = charge_rate / charge_eff
        max_discharge_input = discharge_rate / discharge_eff

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
                actual_charge = max(0.0, min(headroom, max_charge_input, -cmd)) * charge_eff

                # Update SOC (actual_charge is in post-efficiency units)
                soc += actual_charge / storage_capacity
                storage_commodity_out_timesteps[t] = -actual_charge
            else:
                # --- Discharging ---
                # headroom: how much commodity can still be drawn before
                # hitting the minimum SOC, expressed as a rate.
                headroom = (soc - soc_min) * storage_capacity / self.dt_hr

                # Clip and apply discharge efficiency.
                actual_discharge = max(0.0, min(headroom, max_discharge_input, cmd)) * discharge_eff

                # Update SOC (actual_discharge is in post-efficiency units)
                soc -= actual_discharge / storage_capacity
                storage_commodity_out_timesteps[t] = actual_discharge

            soc_timesteps[t] = soc * 100.0

        # Persist the final SOC so subsequent simulate() calls (e.g. from the
        # Pyomo controller across rolling windows) start where we left off.
        self.current_soc = soc
        return storage_commodity_out_timesteps, soc_timesteps
