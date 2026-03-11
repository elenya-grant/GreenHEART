import numpy as np
import pytest
import openmdao.api as om
from pytest import fixture

from h2integrate.storage.simple_storage_auto_sizing import StorageAutoSizingModel


@fixture
def performance_model_config(
    charge_eff, discharge_eff, roundtrip_eff, soc_min, soc_max, charge_equal_discharge
):
    storage_config = {
        "commodity": "hydrogen",
        "commodity_rate_units": "kg/h",
        "max_charge_fraction": soc_max,
        "min_charge_fraction": soc_min,
        "commodity_amount_units": "kg",
        "charge_equals_discharge": charge_equal_discharge,
        "charge_efficiency": charge_eff,
        "discharge_efficiency": discharge_eff,
        "round_trip_efficiency": roundtrip_eff,
        "demand_profile": 0.0,
    }

    return storage_config


@fixture
def plant_config():
    plant = {
        "plant": {
            "plant_life": 30,
            "simulation": {
                "dt": 3600,
                "n_timesteps": 24,
            },
        },
    }
    return plant


@pytest.mark.unit
@pytest.mark.parametrize(
    "case_name,charge_eff,discharge_eff,roundtrip_eff,soc_min,soc_max,charge_equal_discharge",
    [
        ("original", 1.0, 1.0, None, 0.0, 1.0, True),
        ("soc_limited", 1.0, 1.0, None, 0.2, 0.8, True),
        ("charge_losses", 0.8, 0.75, None, 0.0, 1.0, True),
        ("roundtrip_losses", None, None, 0.6, 0.0, 1.0, True),
    ],
    ids=["original", "soc_limited", "charge_losses", "roundtrip_losses"],
)
def test_no_controller_charge_equals_discharge(
    plant_config, performance_model_config, case_name, subtests
):
    # expected_charge, expected_discharge = expected_profiles

    # Get the charge and discharge profiles
    c_eff = (
        performance_model_config["charge_efficiency"]
        if performance_model_config.get("charge_efficiency", None) is not None
        else np.sqrt(performance_model_config["round_trip_efficiency"])
    )
    d_eff = (
        performance_model_config["discharge_efficiency"]
        if performance_model_config.get("discharge_efficiency", None) is not None
        else np.sqrt(performance_model_config["round_trip_efficiency"])
    )

    commodity_in = np.concat([np.zeros(3), np.cumsum(np.ones(15)), np.full(6, 4.0)])

    commodity_demand = np.full(24, np.mean(commodity_in))

    prob = om.Problem()

    commodity_set_point = commodity_demand - commodity_in

    soc_unadjusted_kg = np.cumsum(commodity_in - commodity_demand)
    if np.min(soc_unadjusted_kg) < 0:
        soc_unadjusted_kg = np.abs(np.min(soc_unadjusted_kg)) + soc_unadjusted_kg

    max_storage_capacity = (np.max(soc_unadjusted_kg) - np.min(soc_unadjusted_kg)) / (
        performance_model_config["max_charge_fraction"]
        - performance_model_config["min_charge_fraction"]
    )
    max_charge_rate = np.max(commodity_in)

    prob.model.add_subsystem(
        name="IVC1",
        subsys=om.IndepVarComp(name="hydrogen_in", val=commodity_in, units="kg/h"),
        promotes=["*"],
    )

    prob.model.add_subsystem(
        name="IVC2",
        subsys=om.IndepVarComp(name="hydrogen_demand", val=commodity_demand, units="kg/h"),
        promotes=["*"],
    )

    prob.model.add_subsystem(
        name="IVC3",
        subsys=om.IndepVarComp(name="hydrogen_set_point", val=commodity_set_point, units="kg/h"),
        promotes=["*"],
    )

    prob.model.add_subsystem(
        "storage",
        StorageAutoSizingModel(
            plant_config=plant_config,
            tech_config={"model_inputs": {"performance_parameters": performance_model_config}},
        ),
        promotes=["*"],
    )

    prob.setup()

    prob.run_model()

    charge_rate = prob.get_val("storage.max_charge_rate", units="kg/h")[0]
    discharge_rate = prob.get_val("storage.max_charge_rate", units="kg/h")[0]
    capacity = prob.get_val("storage.max_capacity", units="kg")[0]

    with subtests.test("Charge rate = charge rate from config"):
        assert pytest.approx(charge_rate, rel=1e-6) == max_charge_rate / c_eff
    with subtests.test("Capacity = capacity from config"):
        assert pytest.approx(capacity, rel=1e-6) == max_storage_capacity

    # Test that discharge is always positive
    with subtests.test("Discharge is always positive"):
        assert np.all(prob.get_val("storage.storage_hydrogen_discharge", units="kg/h") >= 0)

    with subtests.test("Charge is always negative"):
        assert np.all(prob.get_val("storage.storage_hydrogen_charge", units="kg/h") <= 0)

    with subtests.test("Charge + Discharge == storage_hydrogen_out"):
        charge_plus_discharge = prob.get_val(
            "storage.storage_hydrogen_charge", units="kg/h"
        ) + prob.get_val("storage.storage_hydrogen_discharge", units="kg/h")
        np.testing.assert_allclose(
            charge_plus_discharge, prob.get_val("storage_hydrogen_out", units="kg/h"), rtol=1e-6
        )

    indx_soc_increase = np.argwhere(
        np.diff(prob.model.get_val("storage.SOC", units="unitless"), prepend=True) > 0
    ).flatten()
    indx_soc_decrease = np.argwhere(
        np.diff(prob.model.get_val("storage.SOC", units="unitless"), prepend=False) < 0
    ).flatten()
    indx_soc_same = np.argwhere(
        np.diff(prob.model.get_val("storage.SOC", units="unitless"), prepend=True) == 0.0
    ).flatten()

    with subtests.test("SOC increases when charging"):
        assert np.all(
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h")[indx_soc_increase] < 0
        )
        assert np.all(
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h")[indx_soc_decrease] == 0
        )
        assert np.all(
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h")[indx_soc_same] == 0
        )

    with subtests.test("SOC decreases when discharging"):
        assert np.all(
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h")[indx_soc_decrease] > 0
        )
        assert np.all(
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h")[indx_soc_increase] == 0
        )
        assert np.all(
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h")[indx_soc_same] == 0
        )

    with subtests.test("Max SOC <= Max storage percent"):
        assert (
            prob.get_val("storage.SOC", units="unitless").max()
            <= performance_model_config["max_charge_fraction"]
        )

    with subtests.test("Min SOC >= Min storage percent"):
        assert (
            prob.get_val("storage.SOC", units="unitless").min()
            >= performance_model_config["min_charge_fraction"]
        )

    with subtests.test("Charge never exceeds charge rate"):
        assert (
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h").min()
            >= -1 * charge_rate * c_eff
        )

    with subtests.test("Discharge never exceeds discharge rate"):
        assert (
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h").max()
            <= discharge_rate * d_eff
        )

    with subtests.test("Discharge never exceeds demand"):
        assert np.all(
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h").max()
            <= commodity_demand
        )

    with subtests.test("Cumulative charge/discharge does not exceed storage capacity"):
        assert np.cumsum(prob.get_val("storage_hydrogen_out", units="kg/h")).max() <= capacity
        assert np.cumsum(prob.get_val("storage_hydrogen_out", units="kg/h")).min() >= -1 * capacity

    with subtests.test("Expected unmet demand"):
        expected_unmet_demand = np.maximum(
            0, commodity_demand - (commodity_in + charge_plus_discharge)
        )
        np.testing.assert_allclose(
            prob.model.get_val("storage.unmet_hydrogen_demand_out", units="kg/h"),
            expected_unmet_demand,
            rtol=1e-6,
            atol=1e-10,
        )
