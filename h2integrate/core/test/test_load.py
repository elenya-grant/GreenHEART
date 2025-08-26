import openmdao.api as om

from h2integrate.core.load_demand import DemandPerformanceModelComponent


def test_remaining_demand():
    plant_config = {"plant": {"simulation": {"n_timesteps": 8760, "dt": 3600}}}
    tech_config = {
        "model_inputs": {
            "performance_parameters": {
                "demand": 100.0,
                "units": "kg",
                "resource_type": "hydrogen",
            }
        }
    }
    prob = om.Problem()
    comp = DemandPerformanceModelComponent(
        plant_config=plant_config,
        tech_config=tech_config,
        driver_config={},
        resource_type="hydrogen",
    )
    prob.model.add_subsystem("comp", comp, promotes=["*"])

    prob.setup()

    prob.set_val("hydrogen_supplied", [50.0])

    prob.run_model()

    assert all(i == 50.0 for i in prob.get_val("hydrogen_demand"))


def test_curtailed_demand():
    plant_config = {"plant": {"simulation": {"n_timesteps": 8760, "dt": 3600}}}
    tech_config = {
        "model_inputs": {
            "performance_parameters": {
                "demand": 100.0,
                "units": "kg",
                "resource_type": "hydrogen",
            }
        }
    }
    prob = om.Problem()
    comp = DemandPerformanceModelComponent(
        plant_config=plant_config,
        tech_config=tech_config,
        driver_config={},
        resource_type="hydrogen",
    )
    prob.model.add_subsystem("comp", comp, promotes=["*"])

    prob.setup()

    prob.set_val("hydrogen_supplied", [150.0])

    prob.run_model()

    assert all(i == 50.0 for i in prob.get_val("hydrogen_curtailed"))
