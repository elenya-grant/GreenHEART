import re

import numpy as np
import pandas as pd
import pytest
import openmdao.api as om
from attrs import field, define, validators

from h2integrate.resource import resource_baseclass
from h2integrate.resource.resource_baseclass import ResourceBaseAPIModel, ResourceBaseAPIConfig


@pytest.fixture
def input_config(
    n_timesteps,
    resource_year,
    include_leap,
    resource_filename,
    resource_filenames,
    resource_year_order,
):
    plant = {
        "plant_life": 30,
        "simulation": {
            "dt": 3600,
            "n_timesteps": n_timesteps,
            "start_time": "01/01/1900 00:30:00",
            "timezone": 0,
        },
    }
    site_config = {
        "latitude": 40.0,
        "longitude": -95.0,
        "resource_year": resource_year,
        "include_leap_day": include_leap,
        "resource_year_order": resource_year_order,
        "resource_filename": resource_filename,
        "resource_filenames": resource_filenames,
        "timezone": 0,
    }

    return {"plant": plant, "site": site_config}


# TODO: add config and class for TMY models
@define(kw_only=True)
class FakeResourceConfig(ResourceBaseAPIConfig):
    resource_year: int = field(converter=int, validator=(validators.ge(2010), validators.le(2020)))
    dataset_desc: str = "fake_api"
    resource_type: str = "fake"
    valid_intervals: list[int] = field(factory=lambda: [30, 60])


class FakeResource(ResourceBaseAPIModel):
    def setup(self):
        resource_specs = self.helper_setup_method()
        self.config = FakeResourceConfig.from_dict(
            resource_specs,
            additional_cls_name=self.__class__.__name__,
        )

        # setup from baseclass
        super().setup()
        self.utc = False
        self.interval = 60  # minutes

        # get the data dictionary
        data = self.get_data(self.config.latitude, self.config.longitude)
        self.resource_data = data

        # add resource data dictionary as an out
        self.add_discrete_output("fake_resource_data", val=data, desc="Dict of fake resource data")

    def get_data_for_year(
        self, latitude, longitude, resource_year, resource_filename="", forced_download=False
    ):
        # simple method that overwrites get_data_for_year in resource baseclass
        filename_match = re.search(r"\d{4}", str(resource_filename))
        if filename_match:
            resource_year = int(filename_match.group(0))
        dates = pd.date_range(
            start=f"{resource_year}-01-01 00:30:00",
            end=f"{resource_year}-12-31 23:30:00",
            freq="1h",
        )

        return {
            "year": dates.year.to_numpy().astype(float),
            "month": dates.month.to_numpy().astype(float),
            "day": dates.day.to_numpy().astype(float),
            "hour": dates.hour.to_numpy().astype(float),
            "minute": dates.hour.to_numpy().astype(float),
            "ws": np.arange(len(dates), dtype=float),
            "latitude": latitude,
            "longitude": longitude,
            "filename": resource_filename,
            "forced_download": forced_download,
            "id": 1111,
            "units": {"ws": "m/s"},
        }

    # def create_filename(self, latitude, longitude, resource_year):
    #     return f"{latitude}_{longitude}_Y{resource_year}_60min_fake.csv"


@pytest.mark.unit
@pytest.mark.parametrize(
    "resource_year,n_timesteps,include_leap,resource_filename,resource_filenames,"
    "resource_year_order,expected_msg",
    [
        (2015, 8760, False, ["file.csv"], None, None, "`resource_filename` must be a single"),
        (2015, 8760, False, "", "file.csv", None, "`resource_filenames` must be a list."),
        (2015, 8760, False, "file.csv", ["file.csv"], None, "Provide either `resource_filename`"),
    ],
    ids=["single-filename-is-list", "filenames-is-not-list", "both-filename-fields"],
)
def test_config_attribute_errors(input_config, expected_msg):
    with pytest.raises(AttributeError) as excinfo:
        FakeResourceConfig.from_dict(input_config["site"])
    assert expected_msg in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "resource_year,n_timesteps,include_leap,resource_filename,resource_filenames,"
    "resource_year_order,expected_msg",
    [
        (2012, 17520, False, "", None, [2012], "2 resource years are required"),
        (2012, 17520, False, "", ["data_2012.csv"], None, "2 resource filenames are required"),
        (
            2012,
            17520,
            False,
            "",
            ["data_2012.csv", "data_2013.csv", "data_2014.csv"],
            None,
            "2 resource filenames are required",
        ),
        (
            2012,
            17520,
            False,
            "",
            ["data_2012.csv", "data_2013.csv"],
            [2012],
            "2 resource years are required",
        ),
    ],
)
def test_setup_errors(input_config, expected_msg):
    prob = om.Problem()
    comp = FakeResource(
        plant_config=input_config,
        resource_config=input_config["site"],
        driver_config={},
    )
    prob.model.add_subsystem("resource", comp)

    with pytest.raises(ValueError) as excinfo:
        prob.setup()
    assert expected_msg in str(excinfo.value)


# @pytest.mark.unit
# def test_check_resource_year(subtests):
#     pass

# @pytest.mark.unit
# def test_get_resource_years_from_start_year(subtests):
#     pass

# @pytest.mark.unit
# def test_process_final_resource_data(subtests):
#     pass

# @pytest.mark.unit
# def test_process_final_resource_data(subtests):
#     pass


@pytest.mark.unit
@pytest.mark.parametrize(
    "resource_year,n_timesteps,include_leap,resource_filename,resource_filenames,resource_year_order",
    [(2012, 17544, True, "", ["data_2013.csv", "data_2012.csv"], None)],
)
def test_get_data_filenames(subtests, input_config):
    prob = om.Problem()
    comp = FakeResource(
        plant_config=input_config,
        resource_config=input_config["site"],
        driver_config={},
    )
    prob.model.add_subsystem("resource", comp)
    prob.setup()
    prob.run_model()

    data_site0 = prob.get_val("resource.fake_resource_data").copy()

    with subtests.test("Initial filename"):
        assert data_site0["filename"] == "data_2012.csv"
    with subtests.test("Years inferred in filename order"):
        assert np.all(data_site0["year"][:8760] == 2013)
        assert np.all(data_site0["year"][8760:] == 2012)

    # Run again, dont change the site
    prob.run_model()

    with subtests.test("Initial filename after rerun"):
        assert prob.get_val("resource.fake_resource_data")["filename"] == "data_2012.csv"

    # Change the site
    prob.set_val("resource.latitude", 35.0, units="deg")
    prob.set_val("resource.longitude", -100.0, units="deg")
    prob.run_model()

    data_site1 = prob.get_val("resource.fake_resource_data").copy()

    with subtests.test("Year order was estimated correctly."):
        assert np.allclose(data_site0["year"], data_site1["year"])

    with subtests.test("Month order was estimated correctly."):
        assert np.allclose(data_site0["month"], data_site1["month"])

    with subtests.test("Latitude changed"):
        assert data_site0["latitude"] != data_site1["latitude"]

    with subtests.test("Longitude changed"):
        assert data_site0["longitude"] != data_site1["longitude"]

    with subtests.test("Filenames changed"):
        assert data_site0["filename"] != data_site1["filename"]

    with subtests.test("Second filename"):
        assert data_site1["filename"] == ""

    with subtests.test("Data length"):
        assert len(data_site1["year"]) == 17544

    # Run again without changing site, make sure filename is still ""
    prob.run_model()
    with subtests.test("Third filename"):
        assert prob.get_val("resource.fake_resource_data")["filename"] == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_timesteps,resource_year,include_leap,resource_filename,resource_filenames,resource_year_order",
    [(17544, 2012, True, "", ["unknown_a.csv", "unknown_b.csv"], None)],
)
def test_filename_year_fallback_on_site_change(input_config, monkeypatch):
    monkeypatch.setattr(resource_baseclass, "estimate_resource_year_from_data", lambda _data: None)

    prob = om.Problem()
    comp = FakeResource(
        plant_config=input_config,
        resource_config=input_config["site"],
        driver_config={},
    )
    prob.model.add_subsystem("resource", comp)
    prob.setup()
    prob.run_model()

    prob.set_val("resource.latitude", 35.0, units="deg")
    prob.set_val("resource.longitude", -100.0, units="deg")
    with pytest.warns(UserWarning, match="using configured resource_year 2012"):
        prob.run_model()

    data_site1 = prob.get_val("resource.fake_resource_data")
    assert np.all(data_site1["year"] == 2012)
