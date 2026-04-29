import networkx as nx
import openmdao.api as om
from attrs import field

from h2integrate.core.utilities import BaseConfig


class SystemControlParameters(BaseConfig):
    draft: str = field()
    solver_name: str = field()
    max_iter: int = field()
    convergence_tolerance: float = field()


class SystemLevelControlDraft(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("driver_config", types=dict)
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)
        self.options.declare("tech_control_types", types=dict)

    def setup(self):
        n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        technology_interconnections = self.options["plant_config"]["technology_interconnections"]
        self.control_config = SystemControlParameters.from_dict(
            self.options["plant_config"]["system_level_control"]["control_parameters"]
        )

        G = nx.DiGraph()
        for connection in technology_interconnections:
            source = connection[0]
            destination = connection[1]
            if len(connection) == 4:
                G.add_edge(source, destination, commodity=connection[2])
            else:
                G.add_edge(source, destination)

        # set([(e[0], e[-1]) for e in G.edges(data="commodity") if len(e) == 3])
        sources_to_commodities = {
            (e[0], e[-1]) for e in G.edges(data="commodity") if e[-1] is not None
        }

        # todo: add logic to prevent including feedstock components
        commodity_to_units = {}
        for src_cmod in sources_to_commodities:
            source_tech, commodity = src_cmod
            if commodity not in commodity_to_units:
                meta_dta = self.add_input(
                    f"{source_tech}_{commodity}_out",
                    shape=n_timesteps,
                    units=None,
                    units_by_conn=True,
                )
                commodity_to_units.update({commodity: meta_dta["units"]})
                self.add_input(
                    f"{source_tech}_rated_{commodity}_production", shape=1, units=meta_dta["units"]
                )
                if self.options["tech_control_types"].get(source_tech, "") == "curtailable":
                    self.add_input(
                        f"{source_tech}_modulated_{commodity}_out",
                        shape=n_timesteps,
                        units=meta_dta["units"],
                    )
            else:
                self.add_input(
                    f"{source_tech}_{commodity}_out",
                    shape=n_timesteps,
                    units=commodity_to_units[commodity],
                )
                self.add_input(
                    f"{source_tech}_rated_{commodity}_production",
                    shape=1,
                    units=commodity_to_units[commodity],
                )
                if self.options["tech_control_types"].get(source_tech, "") == "curtailable":
                    self.add_input(
                        f"{source_tech}_modulated_{commodity}_out",
                        shape=n_timesteps,
                        units=commodity_to_units[commodity],
                    )

        self.commodities_to_units = commodity_to_units
