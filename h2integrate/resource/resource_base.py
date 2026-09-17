import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import openmdao.api as om
from attrs import field, define

from h2integrate.core.utilities import BaseConfig
from h2integrate.core.file_utils import check_resource_dir
from h2integrate.resource.utilities.time_tools import is_leap_year, add_resource_start_end_times
from h2integrate.resource.utilities.download_tools import download_from_api


@define(kw_only=True)
class ResourceBaseAPIConfig(BaseConfig):
    """Base configuration class for resource data downloaded from an API.

    Subclasses should include the following attributes that are not set in this BaseConfig:

        - **resource_year** (*int*): Year to download resource data for.
            Recommended to have a validator for upper and lower limits.
        - **resource_data** (*dict*, optional): Dictionary of user-provided resource data.
            Defaults to {}.
        - **resource_dir** (*str | Path*, optional): Folder to save resource files to or
            load resource files from. Defaults to "".
        - **resource_filename** (*str*, optional): Filename to save resource data to or load
            resource data from. Defaults to None.
        - **valid_intervals** (*list[int]*): time interval(s) in minutes that resource data can be
            downloaded in.

    Note:
        Attributes should be updated in subclasses and should not be modifiable by the user.
        These should be inherit attributes of the subclass.

    Args:
        latitude (float): latitude to download resource data for.
        longitude (float): longitude to download resource data for.
        timezone (float | int): timezone to output data in. May be used to determine whether
            to download data in UTC or local timezone. This should be populated by the value
            in sim_config['timezone']
        resource_data (dict | object, optional): Dictionary of user-input resource data.
            Defaults to an empty dictionary.
        resource_dir (str | Path, optional): Folder to save resource files to or
            load resource files from. Defaults to "".
        resource_filename (str, optional): Filename to save resource data to or load
            resource data from. Defaults to None.

    Attributes:
        dataset_desc (str): description of the dataset, used in file naming.
            Should be updated in a subclass.
        resource_type (str): type of resource data downloaded, used in folder naming.
            Should be updated in a subclass.
    """

    latitude: float = field()
    longitude: float = field()

    timezone: int | float = field()

    dataset_desc: str = field(default="default", init=False)
    resource_type: str = field(default="none", init=False)
    resource_data: dict | object = field(default={})
    resource_filename: Path | str = field(default="")
    resource_dir: Path | str | None = field(default=None)


class ResourceBaseAPIModel(om.ExplicitComponent):
    """Base model for downloading resource data from API calls or loading resource
    data for a single site from a file.

    Attributes
        resource_data (dict | None): resource data that is created in setup() method.
        dt (int): timestep in seconds.
        config (object): configuration class that inherits ResourceBaseAPIConfig.

    Inputs:
        latitude (float): latitude corresponding to location for resource data
        longitude (float): longitude corresponding to location for resource data

    Outputs:
        dict: dictionary of resource data.
    """

    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("resource_config", types=dict)
        self.options.declare("driver_config", types=dict)

    def setup(self):
        # create attributes that will be commonly used for resource classes.
        self.resource_data = None
        self.resource_site = [self.config.latitude, self.config.longitude]
        self.dt = self.options["plant_config"]["plant"]["simulation"]["dt"]
        self.n_timesteps = self.options["plant_config"]["plant"]["simulation"]["n_timesteps"]
        self.add_input("latitude", self.config.latitude, units="deg")
        self.add_input("longitude", self.config.longitude, units="deg")

        self.resource_years = self.get_resource_years()

    def get_resource_years(self):
        resource_year_validator = type(self.config.__attrs_attrs__.resource_year.validator).__name__
        if resource_year_validator == "_InValidator":
            # to accomodate tmy solar resource models
            year_options = self.config.__attrs_attrs__.resource_year.validator.options
            resource_year_type, resource_year = self.config.resource_year.split("-")
            resource_year = int(resource_year)
            self.resource_base_year = deepcopy(resource_year)
            future_years = sorted(
                [
                    int(yr.split("-")[-1])
                    for yr in year_options
                    if (f"{resource_year_type}-" in yr) and int(yr.split("-")[-1]) >= resource_year
                ]
            )

        else:
            for validator in self.config.__attrs_attrs__.resource_year.validator._validators:
                if "<" in validator.compare_op:
                    last_available_yr = (
                        validator.bound
                        if validator.compare_op == "<="
                        else int(validator.bound - 1)
                    )
            future_years = (
                np.arange(self.config.resource_year, last_available_yr + 1, 1).astype(int).tolist()
            )
            self.resource_base_year = deepcopy(self.config.resource_year)

        include_leap = getattr(self.config, "include_leap_day", False)

        if include_leap:
            hours_per_simulation_year = [8784 if is_leap_year(y) else 8760 for y in future_years]
        else:
            hours_per_simulation_year = [8760] * len(future_years)

        future_hours_available = int(sum(hours_per_simulation_year))

        hours_simulated = (self.dt / 3600) * self.n_timesteps

        if future_hours_available < hours_simulated:
            msg = "Not enough future resource years"
            raise ValueError(msg)

        cumulative_hrs = np.cumsum(hours_per_simulation_year)
        last_resource_year = [
            y for y, h in zip(future_years, cumulative_hrs) if h >= hours_simulated
        ][0]

        resource_years = (
            np.arange(self.resource_base_year, last_resource_year + 1, 1).astype(int).tolist()
        )
        if resource_year_validator == "_InValidator":
            resource_years = [f"{resource_year_type}-{int(y)}" for y in resource_years]
        return sorted(resource_years)

    def helper_setup_method(self):
        """
        Prepares and configures resource specifications for the resource API based on plant
        and site configuration options.

        This method extracts relevant configuration details from the `self.options` dictionary,
        pulls values for latitude, longitude, resource directory and timezone from the
        ``site`` section of ``plant_config`` if these parameters are not specified in the
        ``resource_config`` and returns the updated resource specifications dictionary.

        Returns:
            dict: The resource specifications dictionary with defaults set for latitude,
            longitude, resource_dir, and timezone.
        """
        site_config = self.options["plant_config"]["site"]
        sim_config = self.options["plant_config"]["plant"]["simulation"]
        self.dt = sim_config["dt"]

        # create the input dictionary for the resource API config
        resource_specs = self.options["resource_config"]
        # set the default latitude, longitude, and resource_year from the site_config
        resource_specs.setdefault("latitude", site_config["latitude"])
        resource_specs.setdefault("longitude", site_config["longitude"])
        # set the default resource_dir from a directory that can be
        # specified in site_config['resources']['resource_dir']
        resource_specs.setdefault(
            "resource_dir", site_config.get("resources", {}).get("resource_dir", None)
        )

        # default timezone to UTC because 'timezone' was removed from the plant config schema
        resource_specs.setdefault("timezone", sim_config.get("timezone", 0))

        return resource_specs

    def create_filename(self, latitude, longitude):
        """Create default filename to save downloaded data to. Suggested filename formatting is:

        "{latitude}_{longitude}_{resource_year}_{dataset_desc}_{interval}min_{tz_desc}_tz.csv"
        where "tz_desc" is "utc" if the timezone is zero, or "local" otherwise.

        Args:
            latitude (float): latitude corresponding to location for resource data
            longitude (float): longitude corresponding to location for resource data

        Returns:
            str: filename for resource data to be saved to or loaded from.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")

    def create_url(self, latitude, longitude):
        """Create url for data download.

        Args:
            latitude (float): latitude corresponding to location for resource data
            longitude (float): longitude corresponding to location for resource data

        Returns:
            str: url to use for API call.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")

    def download_data(self, url, fpath):
        """Download data from url to a file.

        Args:
            url (str): url to call to access data.
            fpath (Path | str): filepath to save data to.

        Returns:
            bool: True if data was downloaded successfully, False if error was encountered.
        """

        success = download_from_api(url, fpath)
        return success

    def load_data(self, fpath):
        """Loads data from a file, reformats data to follow a standardized naming convention,
        converts data to standardized units, and creates a data time profile.

        Args:
            fpath (str | fpath): filepath to load the data from.

        Raises:
            NotImplementedError: this method should be implemented in a subclass.

        Returns:
            dict: dictionary of data that follows the corresponding standardized
                naming convention and is in standardized units.
                The time profile created should be found in the 'time' key.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def get_data_for_year(self, latitude, longitude, first_call=True):
        """Get resource data to handle any of the expected inputs. This method does the following:

        0) If this is not the first resource call of the simulation, check if latitude and longitude
            inputs are different than the previous latitude and longitude values. If resource data
            has not been already loaded for the, continue to Step 1.
        1) Check if resource data was input. If not, continue to Step 2.
        2) Get valid resource_dir with :py:func:`check_resource_dir`
        3) Create a filename if resource_filename was not input or if the site location changed
            with the method `create_filename()`. Otherwise, use resource_filename as the filename.
        4) If the resulting resource_dir and filename from Steps 2 and 3 make a valid filepath,
            load data using `load_data()`. Otherwise, continue to Step 5.
        5) Create the url to download data using `create_url()` and continue to Step 6.
        6) Download data from the url created in Step 5 and save to a filepath created from the
            resulting resource_dir and filename from Steps 2 and 3. Continue to Step 7.
        7) Load data from the file created in Step 6 using `load_data()`

        Args:
            latitude (float): latitude corresponding to location for resource data
            longitude (float): longitude corresponding to location for resource data
            first_call (bool): True if called from `setup()` method, False if called from
                `compute()` method to prevent unnecessary reloading of data.

        Raises:
            ValueError: If data was not successfully downloaded from the API
            ValueError: An unexpected case was encountered in handling data

        Returns:
            Any: resource data in the format expected by the subclass.
        """
        site_changed = False

        site_changed = not np.allclose([latitude, longitude], self.resource_site, atol=1e-6, rtol=0)

        # 0) If site hasn't changed and resource data has already been loaded
        # just return the resource data that was loaded in the setup() method
        if not site_changed and not first_call:
            if self.resource_data is not None:
                return self.resource_data

        # 1) check if user provided data, add start and end times if so
        # and return the data
        if bool(self.config.resource_data):
            data = add_resource_start_end_times(self.config.resource_data)
            return data

        # check if user provided directory or filename
        provided_filename = False if self.config.resource_filename == "" else True
        provided_dir = False if self.config.resource_dir is None else True

        # 2a) check if file exists directly within resource directory
        # 2) Get valid resource_dir with the function check_resource_dir()
        resource_dir = check_resource_dir(data_dir=self.config.resource_dir)
        # 3a) Create a filename if resource_filename was input
        if provided_filename and not site_changed:
            # If a filename was input, use resource_filename as the filename.
            filepath = resource_dir / self.config.resource_filename
        # Otherwise, create a filename with the method `create_filename()`.
        else:
            filename = self.create_filename(latitude, longitude)
            filepath = resource_dir / filename
        # if file doesn't exist, continue to Step 2b
        if not filepath.is_file():
            # 2b) check if file exists directly within a subfolder of the resource directory
            # 2) Get valid resource_dir with the function check_resource_dir()
            if (
                provided_dir
                and Path(self.config.resource_dir).parts[-1] == self.config.resource_type
            ):
                resource_dir = check_resource_dir(data_dir=self.config.resource_dir)
            else:
                resource_dir = check_resource_dir(
                    data_dir=self.config.resource_dir, data_subdir=self.config.resource_type
                )
            # 3) Create a filename if resource_filename was input
            if provided_filename and not site_changed:
                # If a filename was input, use resource_filename as the filename.
                filepath = resource_dir / self.config.resource_filename
            # Otherwise, create a filename with the method `create_filename()`.
            else:
                filename = self.create_filename(latitude, longitude)
                filepath = resource_dir / filename

        # Check if the filename was provided by the user and the site hasn't changed
        if provided_filename and not site_changed:
            # If the user-provided filename wasn't found, throw a warning
            if not filepath.is_file():
                msg = (
                    f"User provided resource filename {self.config.resource_filename} "
                    f"not found in {resource_dir}. Data will be downloaded for this site."
                )
                warnings.warn(msg, UserWarning)

        # 4) If the resulting resource_dir and filename from Steps 2 and 3 make a valid
        # filepath, load data using `load_data()`
        if filepath.is_file():
            self.filepath = filepath
            data = self.load_data(filepath)
            data = add_resource_start_end_times(data)
            return data

        # If the filepath (resource_dir/filename) does not exist, download data
        self.filepath = filepath
        # 5) Create the url to download data using `create_url()` and continue to Step 6.
        url = self.create_url(latitude, longitude)
        # 6) Download data from the url created in Step 5 and save to a filepath created from
        # the resulting resource_dir and filename from Steps 2 and 3.
        success = self.download_data(url, filepath)
        if success:
            # 7) Load data from the file created in Step 6 using `load_data()`
            data = self.load_data(filepath)
            data = add_resource_start_end_times(data)
            return data

        else:
            raise ValueError("Did not successfully download resource data.")

    def separate_timeseries_and_meta_data(self, data):
        """Separate a dictionary into meta-data and timeseries data components

        Args:
            data (dict): dictionary of resource data

        Returns:
            tuple[dict, dict]: dictionary of meta-data and dictionary of timeseries data
        """
        meta_data = {}
        timeseries_data = {}
        for k, v in data.items():
            if isinstance(v, (str | bool | int | float | dict)):
                meta_data[k] = v
            else:
                timeseries_data[k] = v

        return meta_data, timeseries_data

    def append_timeseries_data(self, ts_data_full, new_ts_data):
        """_summary_

        Args:
            ts_data_full (dict): dictionary of existing timeseries data
            new_ts_data (dict): dictionary of new timeseries data

        Returns:
            dict: dictionary containing timeseries data from ts_data_full and new_ts_data
        """
        shared_keys = set(ts_data_full) & set(new_ts_data)
        if len(shared_keys) != len(set(ts_data_full)):
            # new_ts_data could have extra or new_ts_data could be missing.
            # if shared_keys < len(set(ts_data_full)), then new_ts_data is missing
            missing_data = (set(new_ts_data) - shared_keys) & (set(ts_data_full) - shared_keys)
            msg = (
                f"Mismatch in timeseries data. Non-shared data keys of {sorted(missing_data)} "
                f"will be removed. "
            )
            warnings.warn(msg, UserWarning)

        ts_data = {}
        for k in list(shared_keys):
            if isinstance(ts_data_full[k], list) and isinstance(new_ts_data[k], list):
                ts_data[k] = ts_data_full[k] + new_ts_data[k]
            else:
                ts_data[k] = np.concat(
                    [np.array(ts_data_full[k]), np.array(new_ts_data[k])], axis=0
                )

        return ts_data

    def get_data(self, latitude, longitude, first_call=True):
        site_changed = False

        site_changed = not np.allclose([latitude, longitude], self.resource_site, atol=1e-6, rtol=0)

        # 0) If site hasn't changed and resource data has already been loaded
        # just return the resource data that was loaded in the setup() method
        if not site_changed and not first_call:
            if self.resource_data is not None:
                return self.resource_data

        if len(self.resource_years) == 1:
            resource_data = self.get_data_for_year(latitude, longitude, first_call=first_call)
            return resource_data

        # Multiple years
        if isinstance(self.config.resource_filename, list):
            resource_files = deepcopy(self.config.resource_filename)
        else:
            resource_files = [self.config.resource_filename]

        timeseries_data = {}
        meta_data = {}

        for year in self.resource_years:
            if not isinstance(year, str):
                year = int(year)

            resource_fname_match = [k for k in resource_files if f"_{year}_" in k]
            if bool(resource_fname_match):
                self.config.resource_filename = resource_fname_match[0]

            self.config.resource_year = year

            resource_data = self.get_data_for_year(latitude, longitude, first_call=first_call)
            md, ts = self.separate_timeseries_and_meta_data(resource_data)

            if year == self.resource_years[0]:
                # get start time
                meta_data |= md
                timeseries_data |= ts
            else:
                timeseries_data = self.append_timeseries_data(timeseries_data, ts)

        timeseries_data = add_resource_start_end_times(timeseries_data)
        # reset resource-filename
        self.config.resource_filename = resource_files

        return timeseries_data | meta_data

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # update the resource data based on the input latitude and longitude
        data = self.get_data(inputs["latitude"][0], inputs["longitude"][0], first_call=False)
        # update the stored resource data and site
        self.resource_site = [inputs["latitude"][0], inputs["longitude"][0]]
        self.resource_data = data
        discrete_outputs[f"{self.config.resource_type}_resource_data"] = data
