import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel
from h2integrate.core.inputs.validation import load_tech_yaml, load_plant_yaml, load_driver_yaml


os.chdir(Path(__file__).parent)


# --- Parameters ---
amplitude = 0.9  # Amplitude of the sine wave
frequency = 0.05  # Frequency of the sine wave in Hz
duration = 8760.0  # Duration of the signal in seconds
sampling_rate = 1  # Number of samples per second (Fs)

# Noise parameters
noise_mean = 0.0
noise_std_dev = 0.1  # Standard deviation controls the noise intensity

# --- Generate the Time Vector ---
# Create a time array from 0 to duration with a specific sampling rate
t = np.linspace(1.0, duration, int(sampling_rate * duration), endpoint=True)

# --- Generate the Pure Sine Wave Signal ---
# Formula: y(t) = A * sin(2 * pi * f * t)
pure_signal = amplitude * np.sin(2.0 * np.pi * frequency * t)

# --- Generate the Random Gaussian Noise ---
# Create noise with the same shape as the time vector
rng = np.random.default_rng()
noise = rng.normal(loc=noise_mean, scale=noise_std_dev, size=t.shape)

# --- Create the Noisy Signal ---
noisy_signal = (pure_signal + noise) * 0.04 + 0.04 * np.ones(len(t))

commodity_met_value_profile = np.ones(8760) * 1
commodity_buy_price_profile = noisy_signal

demand_profile = np.ones(8760) * 100.0

# Modify stuff
tech_config = load_tech_yaml("tech_config.yaml")
plant_config = load_plant_yaml("plant_config.yaml")
driver_config = load_driver_yaml("driver_config.yaml")

tech_config["technologies"]["grid_buy"]["model_inputs"]["cost_parameters"][
    "electricity_buy_price"
] = commodity_buy_price_profile

config = {
    "plant_config": plant_config,
    "technology_config": tech_config,
    "driver_config": driver_config,
}

# Create an H2Integrate model
model = H2IntegrateModel(config)


# TODO: Update with demand module once it is developed
model.setup()
model.prob.set_val("battery.electricity_demand", demand_profile, units="MW")
model.prob.set_val("real_battery.electricity_demand", demand_profile, units="MW")
# model.prob.set_val("battery.demand_met_value", commodity_met_value_profile, units="USD/kW")
model.prob.set_val("battery.electricity_buy_price", commodity_buy_price_profile, units="USD/kW")
# model.prob.set_val("electricity_feedstock.price", commodity_buy_price_profile, units="USD/(kW*h)")
model.prob.set_val(
    "grid_buy.electricity_buy_price", commodity_buy_price_profile, units="USD/(kW*h)"
)


# Run the model
model.run()

# model.post_process()

# Elenya: checking logic
controller_soc = model.prob.get_val("battery.controller_estimated_SOC", units="percent")
dummy_SOC = model.prob.get_val("battery.SOC", units="percent")
actual_SOC = model.prob.get_val("real_battery.SOC", units="percent")

# below should be zero because we can buy all the power from the grid
soc_error_real_battery = (np.abs(controller_soc - actual_SOC)).sum()

# below should be nonzero but isn't?
soc_error_dummy_battery = (np.abs(controller_soc - dummy_SOC)).sum()

# check that dummy battery is not charging with power that isn't there
electricity_in = model.prob.get_val("battery.electricity_in", units="kW")

# Electricity from grid to battery
electricity_to_charge_battery = model.prob.get_val(
    "battery.electricity_bought_for_storage", units="kW"
)

battery_charge_profile = model.prob.get_val("battery.storage_electricity_charge", units="kW")

combined_electricity_out = electricity_in + model.prob.get_val("battery.storage_electricity_out")

unused_electricity = np.where(
    combined_electricity_out > model.prob.get_val("battery.electricity_demand", units="kW"),
    combined_electricity_out - model.prob.get_val("battery.electricity_demand", units="kW"),
    0,
)

electricity_in.sum() + model.prob.get_val("battery.unused_electricity_out", units="kW").sum()

model.prob.get_val("battery.electricity_out", units="kW").sum() - ()

only_buy_from_grid = electricity_to_charge_battery.min() >= 0.0


# Plot the results
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

start_hour = 0
end_hour = 200

ax[0].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.SOC", units="percent")[start_hour:end_hour],
    label="SOC",
)
ax[0].set_ylabel("SOC (%)")
ax[0].set_ylim([0, 110])
ax[0].axhline(y=90.0, linestyle=":", color="k", alpha=0.5, label="Max Charge")
ax[0].legend()

ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_in", units="MW")[start_hour:end_hour],
    linestyle="-",
    label="Electricity In (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.unused_electricity_out", units="MW")[start_hour:end_hour],
    linestyle=":",
    label="Unused Electricity (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.unmet_electricity_demand_out", units="MW")[start_hour:end_hour],
    linestyle=":",
    label="Unmet Electrical Demand (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_out", units="MW")[start_hour:end_hour],
    linestyle="-",
    label="Electricity Out (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_bought_for_storage", units="MW")[start_hour:end_hour],
    linestyle="-",
    label="Electricity Bought (MW)",
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.battery_electricity_out", units="MW")[start_hour:end_hour],
    linestyle="-.",
    label="Battery Electricity Out (MW)",
)

print(min(model.prob.get_val("battery.electricity_out", units="MW")))
print(min(model.prob.get_val("battery.battery_electricity_out", units="MW")))
ax[1].plot(
    range(start_hour, end_hour),
    demand_profile[start_hour:end_hour],
    linestyle="--",
    label="Electrical Demand (MW)",
)
ax[1].set_ylim([-1e2, 2.5e2])
ax[1].set_ylabel("Electricity Hourly (MW)")
ax[1].legend()

ax[2].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_buy_price", units="USD/MW")[start_hour:end_hour],
    label="Grid Purchase Price ($/MW)",
)
ax[2].set_ylabel("Grid Purchase Price ($/MW)")
ax[2].set_xlabel("Timestep (hr)")

plt.legend(ncol=2, frameon=False)
plt.tight_layout()
plt.savefig("optimized_dispatch_plot.png", dpi=300)
