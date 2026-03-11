import numpy as np
import pytest
import openmdao.api as om
from pytest import fixture

from h2integrate.storage.simple_generic_storage import SimpleGenericStorage


@fixture
def performance_model_config(
    charge_eff, discharge_eff, roundtrip_eff, soc_min, soc_max, discharge_rate, demand_profile
):
    storage_config = {
        "commodity": "hydrogen",
        "commodity_rate_units": "kg/h",
        "max_charge_rate": 10.0,
        "max_capacity": 40.0,
        "max_charge_fraction": soc_max,
        "min_charge_fraction": soc_min,
        "init_charge_fraction": 0.1,
        "commodity_amount_units": "kg",
        "max_discharge_rate": discharge_rate,
        "charge_equals_discharge": True,
        "charge_efficiency": charge_eff,
        "discharge_efficiency": discharge_eff,
        "round_trip_efficiency": roundtrip_eff,
        "demand_profile": demand_profile,
    }

    if discharge_rate != storage_config["max_charge_rate"]:
        storage_config["charge_equals_discharge"] = False

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


@fixture
def expected_profiles(case_name, discharge_eff, roundtrip_eff):
    match case_name:
        case "charge>demand":
            expected_charge = np.concat([np.zeros(8), np.arange(-1, -9, -1), np.zeros(8)])
            expected_discharge = np.concat([np.zeros(18), np.ones(6)])

        case "charge<demand":
            expected_charge = np.concat(
                [np.full(3, -9), np.zeros(11), np.arange(-1, -5, -1), np.zeros(6)]
            )
            expected_discharge = np.concat(
                [np.zeros(3), np.array([10, 9, 8]), np.zeros(12), np.array([7, 3]), np.zeros(4)]
            )

        case "losses":
            expected_charge = np.concat(
                [np.zeros(8), np.arange(-0.8, -7.2, -0.8), np.array([-5.76, -1.152]), np.zeros(6)]
            )
            expected_discharge = np.concat([np.zeros(18), discharge_eff * np.ones(6)])

        case "roundtrip_losses":
            expected_charge = np.concat(
                [
                    np.zeros(8),
                    np.arange(
                        -1 * np.sqrt(roundtrip_eff),
                        -6.5,
                        -np.sqrt(roundtrip_eff),
                    ),
                    np.array([-6.28548009, -1.41676815]),
                    np.zeros(6),
                ]
            )
            expected_discharge = np.concat([np.zeros(18), np.sqrt(roundtrip_eff) * np.ones(6)])

    return (expected_charge, expected_discharge)


@pytest.mark.unit
@pytest.mark.parametrize(
    "case_name,charge_eff,discharge_eff,roundtrip_eff,soc_min,soc_max,discharge_rate,demand_profile",
    [
        ("charge>demand", 1.0, 1.0, None, 0.1, 1.0, 10.0, 5.0),
        ("charge<demand", 1.0, 1.0, None, 0.1, 1.0, 10.0, 11.0),
        ("losses", 0.8, 0.75, None, 0.1, 1.0, 10.0, 5.0),
        ("roundtrip_losses", None, None, 0.6, 0.1, 1.0, 10.0, 5.0),
    ],
    ids=["charge>demand", "charge<demand", "losses", "roundtrip_losses"],
)
def test_no_controller_charge_equals_discharge(
    plant_config, performance_model_config, expected_profiles, case_name, subtests
):
    expected_charge, expected_discharge = expected_profiles

    commodity_demand = np.full(24, performance_model_config["demand_profile"])

    cases_start_with_zero_in = ["losses", "charge>demand", "roundtrip_losses"]
    if case_name in cases_start_with_zero_in:
        commodity_in = np.concat([np.zeros(3), np.cumsum(np.ones(15)), np.full(6, 4.0)])
    else:
        commodity_in = np.concat([np.full(3, 20.0), np.cumsum(np.ones(15)), np.full(6, 4.0)])

    prob = om.Problem()

    commodity_set_point = commodity_demand - commodity_in

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
        SimpleGenericStorage(
            plant_config=plant_config,
            tech_config={"model_inputs": {"performance_parameters": performance_model_config}},
        ),
        promotes=["*"],
    )

    prob.setup()

    prob.run_model()

    charge_rate = prob.get_val("storage.max_charge_rate", units="kg/h")[0]
    discharge_rate = prob.get_val("storage.max_charge_rate", units="kg/h")[0]
    capacity = prob.get_val("storage.storage_capacity", units="kg")[0]

    with subtests.test("Charge rate = charge rate from config"):
        assert pytest.approx(charge_rate, rel=1e-6) == performance_model_config["max_charge_rate"]
    with subtests.test("Capacity = capacity from config"):
        assert pytest.approx(capacity, rel=1e-6) == performance_model_config["max_capacity"]

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
    with subtests.test("Initial SOC is correct"):
        init_soc_expected = (
            performance_model_config["init_charge_fraction"]
            - prob.get_val("storage_hydrogen_out", units="kg/h")[0] / capacity
        )
        assert (
            pytest.approx(prob.model.get_val("storage.SOC", units="unitless")[0], rel=1e-6)
            == init_soc_expected
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
        c_eff = (
            performance_model_config["charge_efficiency"]
            if performance_model_config.get("charge_efficiency", None) is not None
            else np.sqrt(performance_model_config["round_trip_efficiency"])
        )
        assert (
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h").min()
            >= -1 * charge_rate * c_eff
        )

    with subtests.test("Discharge never exceeds discharge rate"):
        d_eff = (
            performance_model_config["discharge_efficiency"]
            if performance_model_config.get("discharge_efficiency", None) is not None
            else np.sqrt(performance_model_config["round_trip_efficiency"])
        )
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

    with subtests.test("Expected discharge"):
        np.testing.assert_allclose(
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h"),
            expected_discharge,
            rtol=1e-6,
            atol=1e-10,
        )

    with subtests.test("Expected charge"):
        np.testing.assert_allclose(
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h"),
            expected_charge,
            rtol=1e-6,
            atol=1e-10,
        )
