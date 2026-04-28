class ControllerHCST:
    """Controller implementing the HCST schedule described in the module docstring."""

    def __init__(self, h_dict, component_name="hard_coal_steam_turbine"):
        """Initialize the controller.

        Args:
            h_dict (dict): The hercules input dictionary.

        """
        self.component_name = component_name
        self.rated_capacity = h_dict[self.component_name]["rated_capacity"]

        simulation_length = h_dict["endtime_utc"] - h_dict["starttime_utc"]
        self.total_simulation_time = simulation_length.total_seconds()

    def step(self, h_dict):
        """Execute one control step.
        This controller is scaled by the total simulation time, pulled from the h_dict
        This preserves the relative distance between control actions, but changes the
            simulation times that they are applied.

        Args:
            h_dict (dict): The hercules input dictionary.

        Returns:
            dict: The updated hercules input dictionary.

        """
        current_time = h_dict["time"]

        # Determine power setpoint based on time
        if current_time < 0.05 * self.total_simulation_time:
            # First 5% of simulation time, run at full capacity
            power_setpoint = self.rated_capacity
        elif current_time < 0.15 * self.total_simulation_time:
            # Between 5% and 15% of simulation time: shut down
            power_setpoint = 0.0
        elif current_time < 0.45 * self.total_simulation_time:
            # Between 15% and 45% of simulation time: signal to run at full capacity
            power_setpoint = self.rated_capacity
        elif current_time < 0.65 * self.total_simulation_time:
            # Between 45% and 65% of simulation time: reduce power to 50% of rated capacity
            power_setpoint = 0.5 * self.rated_capacity
        elif current_time < 0.75 * self.total_simulation_time:
            # Between 65% and 75% of simulation time: reduce power to 10% of rated capacity
            power_setpoint = 0.1 * self.rated_capacity
        elif current_time < 0.9 * self.total_simulation_time:  #
            # Between 75% and 90% of simulation time: increase power to 100% of rated capacity
            power_setpoint = self.rated_capacity
        else:
            # After 90% of simulation time: shut down
            power_setpoint = 0.0

        h_dict[self.component_name]["power_setpoint"] = power_setpoint

        return h_dict


class ControllerOCGT:
    """Controller implementing the OCGT schedule described in the module docstring."""

    def __init__(self, h_dict, component_name="open_cycle_gas_turbine"):
        """Initialize the controller.

        Args:
            h_dict (dict): The hercules input dictionary.

        """
        self.component_name = component_name
        self.rated_capacity = h_dict[self.component_name]["rated_capacity"]

    def step(self, h_dict):
        """Execute one control step.

        Args:
            h_dict (dict): The hercules input dictionary.

        Returns:
            dict: The updated hercules input dictionary.

        """
        current_time = h_dict["time"]

        # Determine power setpoint based on time
        if current_time < 10 * 60:  # 10 minutes in seconds
            # Before 10 minutes: run at full capacity
            power_setpoint = self.rated_capacity
        elif current_time < 40 * 60:  # 40 minutes in seconds
            # Between 10 and 40 minutes: shut down
            power_setpoint = 0.0
        elif current_time < 120 * 60:  # 120 minutes in seconds
            # Between 40 and 120 minutes: signal to run at full capacity
            power_setpoint = self.rated_capacity
        elif current_time < 180 * 60:  # 180 minutes in seconds
            # Between 120 and 180 minutes: reduce power to 50% of rated capacity
            power_setpoint = 0.5 * self.rated_capacity
        elif current_time < 210 * 60:  # 210 minutes in seconds
            # Between 180 and 210 minutes: reduce power to 10% of rated capacity
            power_setpoint = 0.1 * self.rated_capacity
        elif current_time < 240 * 60:  # 240 minutes in seconds
            # Between 210 and 240 minutes: increase power to 100% of rated capacity
            power_setpoint = self.rated_capacity
        else:
            # After 240 minutes: shut down
            power_setpoint = 0.0

        h_dict[self.component_name]["power_setpoint"] = power_setpoint

        return h_dict
