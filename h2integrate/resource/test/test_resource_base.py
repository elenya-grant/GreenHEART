import pytest
import openmdao.api as om

from h2integrate.resource.solar.nlr_developer_goes_api_models import GOESAggregatedSolarAPI


# docs fencepost start: DO NOT REMOVE
# fmt: off
@pytest.mark.unit
@pytest.mark.parametrize(
    "model,which,lat,lon,resource_year,model_name,timezone",
    [("GOESAggregatedSolarAPI", "solar", 34.22, -102.75, 2012, "goes_aggregated_v4", 0)],
    ids=["GOESAggregatedSolarAPI"]
)
# fmt: on
def test_changing_resource_site_with_filename(
    subtests,
    plant_simulation,
    site_config,
    model,
    which,
    lat,
    lon,
    resource_year,
    model_name,
):

    site_config["resources"]["solar_resource"]["resource_parameters"].update({"resource_filename":"34.22_-102.75_2012_goes_aggregated_v4_60min_utc_tz.csv"})
    plant_config = {
        "site": site_config,
        "plant": plant_simulation,
    }


    sites_to_expected_site_data = {
        (34.22, -102.75): (34.21, -102.74, 542970),
        (35.2018863, -101.945027): (35.21, -101.94, 564069)
    }

    prob = om.Problem()
    comp = GOESAggregatedSolarAPI(
        plant_config=plant_config,
        resource_config=plant_config["site"]["resources"]["solar_resource"]["resource_parameters"],
        driver_config={},
    )
    prob.model.add_subsystem("resource", comp)
    prob.setup()

    # run 0: using site (34.21, -102.74)
    prob.run_model()
    prob.get_val("resource.solar_resource_data")
    result_data_site = (
        prob.get_val("resource.solar_resource_data")["site_lat"],
        prob.get_val("resource.solar_resource_data")["site_lon"],
        prob.get_val("resource.solar_resource_data")["id"]
    )
    current_site = (
        prob.get_val("resource.latitude", units="deg")[0],
        prob.get_val("resource.longitude", units="deg")[0]
        )

    with subtests.test("Run 0: using resource data for site0"):
        assert sites_to_expected_site_data[current_site] == result_data_site

    # Run 1: change latitude and longitude to the other site
    prob.set_val("resource.latitude", 35.2018863, units="deg")
    prob.set_val("resource.longitude", -101.945027, units="deg")
    prob.run_model()
    result_data_site1 = (
        prob.get_val("resource.solar_resource_data")["site_lat"],
        prob.get_val("resource.solar_resource_data")["site_lon"],
        prob.get_val("resource.solar_resource_data")["id"]
    )
    current_site1 = (
        prob.get_val("resource.latitude", units="deg")[0],
        prob.get_val("resource.longitude", units="deg")[0]
        )
    with subtests.test("Run 1: using resource data for changed site (site_changed=True)"):
        assert sites_to_expected_site_data[current_site1] == result_data_site1

    # Run 2: don't change location, and re-run
    prob.run_model()
    result_data_site2 = (
        prob.get_val("resource.solar_resource_data")["site_lat"],
        prob.get_val("resource.solar_resource_data")["site_lon"],
        prob.get_val("resource.solar_resource_data")["id"]
    )
    current_site2 = (
        prob.get_val("resource.latitude", units="deg")[0],
        prob.get_val("resource.longitude", units="deg")[0]
        )
    with subtests.test("Run 2: using resource data for changed site with site_change=False"):
        assert sites_to_expected_site_data[current_site2] == result_data_site2
