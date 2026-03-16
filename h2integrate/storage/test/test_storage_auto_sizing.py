import numpy as np
import pytest
import openmdao.api as om

from h2integrate.storage.simple_storage_auto_sizing import StorageAutoSizingModel


@pytest.mark.regression
@pytest.mark.parametrize("n_timesteps", [24])
def test_storage_autosizing_basic_performance_no_losses(plant_config, subtests):
    # Basic test to ensure that storage outputs (charge profile, discharge profile, SOC)
    # don't violate any performance constraints and that the calculated storage sizes
    # are as expected
    performance_model_config = {
        "commodity": "hydrogen",
        "commodity_rate_units": "kg/h",
        "set_demand_as_avg_commodity_in": True,
        "min_charge_fraction": 0.0,
        "max_charge_fraction": 1.0,
        "commodity_amount_units": "kg",
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
    }

    commodity_in = np.concat(
        [np.full(3, 12.0), np.cumsum(np.ones(15)), np.full(3, 4.0), np.zeros(3)]
    )
    commodity_demand = np.full(24, np.mean(commodity_in))

    prob = om.Problem()

    prob.model.add_subsystem(
        name="IVC1",
        subsys=om.IndepVarComp(name="hydrogen_in", val=commodity_in, units="kg/h"),
        promotes=["*"],
    )

    prob.model.add_subsystem(
        name="IVC3",
        subsys=om.IndepVarComp(
            name="hydrogen_set_point", val=commodity_demand - commodity_in, units="kg/h"
        ),
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
    capacity = prob.get_val("storage.storage_capacity", units="kg")[0]

    with subtests.test("Charge rate value"):
        assert pytest.approx(charge_rate, rel=1e-6) == np.max(commodity_in)

    with subtests.test("Storage capacity value"):
        soc_kg = np.cumsum(commodity_demand - commodity_in)
        soc_kg_adj = soc_kg + np.abs(np.min(soc_kg))
        expected_capacity = np.max(soc_kg_adj) - np.min(soc_kg_adj)
        assert pytest.approx(capacity, rel=1e-6) == expected_capacity

    # TODO: add test for storage duration
    with subtests.test("Storage duration"):
        expected_storage_duration = expected_capacity / np.max(commodity_in)
        assert (
            pytest.approx(prob.get_val("storage_duration", units="h"), rel=1e-6)
            == expected_storage_duration
        )

    # Basic sanity-check unit tests on storage performance
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

    # Check that never charging and discharging at the same time
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

    # Check that charge rate limits are respected
    with subtests.test("Charge never exceeds charge rate"):
        assert (
            prob.get_val("storage.storage_hydrogen_charge", units="kg/h").min() >= -1 * charge_rate
        )

    with subtests.test("Discharge never exceeds discharge rate"):
        assert prob.get_val("storage.storage_hydrogen_discharge", units="kg/h").max() <= charge_rate

    with subtests.test("Discharge never exceeds demand"):
        assert np.all(
            prob.get_val("storage.storage_hydrogen_discharge", units="kg/h").max()
            <= commodity_demand
        )

    with subtests.test("Cumulative charge/discharge does not exceed storage capacity"):
        assert np.cumsum(prob.get_val("storage_hydrogen_out", units="kg/h")).max() <= capacity
        assert np.cumsum(prob.get_val("storage_hydrogen_out", units="kg/h")).min() >= -1 * capacity

    # Check that demand is fully met, this is because this test starts off with charging the storage
    # enough. In cases where the storage is not charged enough at the start, the demand may not
    # fully be met
    with subtests.test("Demand is fully met"):
        np.testing.assert_allclose(
            prob.get_val("hydrogen_out", units="kg/h"), commodity_demand, rtol=1e-6, atol=1e-10
        )

    # TODO: add subtests for unmet demand, and excess commodity, etc


@pytest.mark.regression
@pytest.mark.parametrize("n_timesteps", [24])
def test_storage_autosizing_soc_bounds(plant_config, subtests):
    pass


@pytest.mark.regression
@pytest.mark.parametrize("n_timesteps", [24])
def test_storage_autosizing_charge_losses(plant_config, subtests):
    pass
