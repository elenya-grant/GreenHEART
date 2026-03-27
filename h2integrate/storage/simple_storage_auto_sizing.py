import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


@define(kw_only=True)
class StorageSizingModelConfig(BaseConfig):
    """Configuration class for the StorageAutoSizingModel.

    Fields include `commodity`, `commodity_rate_units`, and `demand_profile`.
    """

    commodity: str = field(default="hydrogen")
    commodity_rate_units: str = field(default="kg/h")  # TODO: update to commodity_rate_units
    demand_profile: int | float | list = field(default=0.0)


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
            the simulation. Defined in `commodity_rate_units*h`
        rated_{commodity}_production (float): maximum commodity that could be discharged
            in a timestep. Defined in `commodity_rate_units`
        annual_{commodity}_produced (np.ndarray): total commodity discharged per year.
            Defined in `commodity_rate_units*h/year`
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
        self.commodity_amount_units = f"({self.commodity_rate_units})*h"

        super().setup()

        self.add_input(
            f"{self.commodity}_demand",
            units=f"{self.config.commodity_rate_units}",
            val=self.config.demand_profile,
            shape=self.n_timesteps,
            desc=f"{self.commodity} demand profile timeseries",
        )

        self.add_input(
            f"{self.commodity}_in",
            shape_by_conn=True,
            units=f"{self.config.commodity_rate_units}",
            desc=f"{self.commodity} input timeseries from production to storage",
        )

        self.add_input(
            f"{self.commodity}_set_point",
            shape_by_conn=True,
            units=f"{self.config.commodity_rate_units}",
            desc=f"{self.commodity} input set point from controller",
        )

        self.add_output(
            "max_capacity",
            val=0.0,
            shape=1,
            units=f"({self.config.commodity_rate_units})*h",
        )

        self.add_output(
            "max_charge_rate",
            val=0.0,
            shape=1,
            units=f"{self.config.commodity_rate_units}",
        )

        self.dt_hr = int(self.options["plant_config"]["plant"]["simulation"]["dt"]) / (
            60**2
        )  # convert from seconds to hours

    def compute(self, inputs, outputs):
        # Step 1: Auto-size the storage to meet the demand

        # Auto-size the fill rate as the max of the input commodity
        storage_max_fill_rate = np.max(inputs[f"{self.commodity}_in"])

        # Set the demand profile
        if np.sum(inputs[f"{self.commodity}_demand"]) > 0:
            commodity_demand = inputs[f"{self.commodity}_demand"]
        else:
            # If the commodity_demand is zero, use the average
            # commodity_in as the demand
            commodity_demand = np.mean(inputs[f"{self.commodity}_in"]) * np.ones(
                self.n_timesteps
            )  # TODO: update demand based on end-use needs

        # The commodity_set_point is the production set by the controller
        # storage_dispatch_commands = inputs[f"{self.commodity}_set_point"]

        # TODO: SOC is just an absolute value and is not a percentage. Ideally would calculate as shortfall in future.
        # Size the storage capacity to meet the demand as much as possible
        commodity_storage_soc = []
        for j in range(len(inputs[f"{self.commodity}_in"])):
            if j == 0:
                commodity_storage_soc.append(
                    inputs[f"{self.commodity}_in"][j] - commodity_demand[j]
                )
            else:
                commodity_storage_soc.append(
                    commodity_storage_soc[j - 1]
                    + inputs[f"{self.commodity}_in"][j]
                    - commodity_demand[j]
                )

        minimum_soc = np.min(commodity_storage_soc)

        # Adjust soc so it's not negative.
        if minimum_soc < 0:
            commodity_storage_soc = [x + np.abs(minimum_soc) for x in commodity_storage_soc]

        # Calculate the maximum hydrogen storage capacity needed to meet the demand
        commodity_storage_capacity = np.max(commodity_storage_soc) - np.min(commodity_storage_soc)

        # Step 2: Simulate the storage performance based on the sizes calculated
        self.current_soc = commodity_storage_soc[0] / commodity_storage_capacity

        storage_commodity_out, soc = self.simulate(
            storage_dispatch_commands=inputs[f"{self.commodity}_set_point"],
            charge_rate=storage_max_fill_rate,
            discharge_rate=storage_max_fill_rate,
            storage_capacity=commodity_storage_capacity,
        )

        # determine storage charge and discharge
        # storage_commodity_out is positive when the storage is discharged
        # and negative when the storage is charged
        storage_commodity_out = np.array(storage_commodity_out)

        # calculate combined commodity out from inflow source and storage
        # (note: storage_commodity_out is negative when charging)
        combined_commodity_out = inputs[f"{self.commodity}_in"] + storage_commodity_out

        # find the total commodity out to meet demand
        total_commodity_out = np.minimum(commodity_demand, combined_commodity_out)

        # determine how much of the inflow commodity was unused
        # unused_commodity = np.maximum(
        #     0, combined_commodity_out - inputs[f"{self.commodity}_demand"]
        # )

        # # determine how much demand was not met
        # unmet_demand = np.maximum(
        #     0, inputs[f"{self.commodity}_demand"] - combined_commodity_out
        # )

        discharge_storage = np.where(storage_commodity_out > 0, storage_commodity_out, 0)

        # Output the storage sizes (charge rate and capacity)
        outputs["max_charge_rate"] = storage_max_fill_rate
        outputs["max_capacity"] = commodity_storage_capacity

        # commodity_out is the commodity_set_point - charge_storage + discharge_storage
        outputs[f"{self.commodity}_out"] = total_commodity_out

        # The rated_commodity_production is based on the discharge rate
        # (which is assumed equal to the charge rate)
        outputs[f"rated_{self.commodity}_production"] = storage_max_fill_rate

        # The total_commodity_produced is the sum of the commodity discharged from storage
        outputs[f"total_{self.commodity}_produced"] = discharge_storage.sum()
        # Adjust the total_commodity_produced to a year-long simulation
        outputs[f"annual_{self.commodity}_produced"] = outputs[
            f"total_{self.commodity}_produced"
        ] * (1 / self.fraction_of_year_simulated)

        # The maximum production is based on the charge/discharge rate
        max_production = storage_max_fill_rate * self.n_timesteps * (self.dt / 3600)

        # Capacity factor is total discharged commodity / maximum discharged commodity possible
        outputs["capacity_factor"] = outputs[f"total_{self.commodity}_produced"] / max_production

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
        charge_eff = 1.0
        discharge_eff = 1.0
        soc_max = 1.0
        soc_min = 0.0

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
