import numpy as np
import pytest
import openmdao.api as om

from h2integrate.storage.generic_storage_pyo import StoragePerformanceModel


@pytest.mark.regression
@pytest.mark.parametrize("n_timesteps", [24])
def test_generic_storage_without_controller(plant_config, subtests):
    performance_model_config = {
        "commodity": "hydrogen",
        "commodity_rate_units": "kg/h",
        "max_capacity": 40,
        "max_charge_rate": 10,
        "min_charge_percent": 0.1,
        "max_charge_percent": 1.0,
        "init_charge_percent": 0.1,
        "n_control_window": 24,
        "commodity_amount_units": "kg",
        "charge_equals_discharge": True,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
    }

    prob = om.Problem()

    commodity_demand = np.full(24, 5.0)
    commodity_in = np.concat([np.zeros(3), np.cumsum(np.ones(15)), np.full(6, 4.0)])

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
        "storage",
        StoragePerformanceModel(
            plant_config=plant_config,
            tech_config={"model_inputs": {"performance_parameters": performance_model_config}},
        ),
        promotes=["*"],
    )

    prob.setup()

    prob.run_model()

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
        assert (
            pytest.approx(prob.model.get_val("storage.SOC", units="unitless")[0], rel=1e-6) == 0.1
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
        assert prob.get_val("storage.SOC", units="unitless").max() <= 1.0

    with subtests.test("Min SOC >= Min storage percent"):
        assert prob.get_val("storage.SOC", units="unitless").min() >= 0.1

    with subtests.test("Charge never exceeds charge rate"):
        assert prob.get_val("storage.storage_hydrogen_charge", units="kg/h").min() >= -10

    with subtests.test("Discharge never exceeds discharge rate"):
        assert prob.get_val("storage.storage_hydrogen_discharge", units="kg/h").max() <= 10.0

    with subtests.test("Cumulative charge/discharge does not exceed storage capacity"):
        assert np.cumsum(prob.get_val("storage_hydrogen_out", units="kg/h")).max() <= 40.0
        assert np.cumsum(prob.get_val("storage_hydrogen_out", units="kg/h")).min() >= -40.0

    with subtests.test("Expected discharge"):
        expected_discharge = np.concat([np.zeros(18), np.ones(6)])
        np.testing.assert_allclose(
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h"),
            expected_discharge,
            rtol=1e-6,
        )

    with subtests.test("Expected charge"):
        expected_charge = np.concat([np.zeros(8), np.arange(-1, -9, -1), np.zeros(8)])
        np.testing.assert_allclose(
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h"),
            expected_charge,
            rtol=1e-6,
        )
