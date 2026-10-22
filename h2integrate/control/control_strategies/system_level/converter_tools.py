def get_successors_for_tech_with_input_cmod(technology_graph, tech, input_commodity):
    """Find technologies upstream of ``tech`` that produce ``input_commodity``
    for ``tech``.

    Args:
        tech (str): Technology whose upstream suppliers are sought.
        commodity (str): Commodity of interest that is an input commodity to ``tech``
            (e.g. ``"electricity"``).

    Returns:
        list[str]: Controller-managed technologies upstream of ``tech``
        that produce ``commodity``.
    """
    in_flows = dict(technology_graph.in_degree)
    if in_flows[tech] < 1:
        # Tech does not have any input commodiites
        return []

    successor_techs_with_commod = set()
    upstream_techs = set(technology_graph.predecessors(tech))
    for upstream_tech in upstream_techs:
        produces_cmod = False
        if (commod := technology_graph.edges[upstream_tech, tech].get("commodity")) is not None:
            if isinstance(commod, str) and commod == input_commodity:
                # this if-statement is outdated and could be removed
                successor_techs_with_commod.add(upstream_tech)
                produces_cmod = True
            if isinstance(commod, list) and input_commodity in commod:
                successor_techs_with_commod.add(upstream_tech)
                produces_cmod = True
        if in_flows[upstream_tech] > 1 and produces_cmod:
            # if only use >1, then it wouldn't catch splitters
            # use `in_flows[upstream_tech] >= 1` to properly handle splitters
            new_techs = get_successors_for_tech_with_input_cmod(
                technology_graph, upstream_tech, input_commodity
            )
            if new_techs:
                successor_techs_with_commod |= set(new_techs)

    return list(successor_techs_with_commod)


def _find_converter_techs(technology_graph, input_techs, storage_techs):
    """Identify technologies that transform one commodity into another.

    A "converter" is a tech whose output commodities differ from the commodities
    produced by its upstream ancestors (e.g. an electrolyzer: electricity → hydrogen).

    Returns:
        tuple[set, dict]: 2-element tuple containing ``converters`` and ``converter_upstreams``.

        **converters** *(set[tuple])*: Set of tuples formatted as
        ``(input_commodity, tech_name, output_commodity)`` tuples.

            >>> converters  # tuples formatted as (input_commodity, tech_name, output_commodity)
            {
                # (input_commodity, tech_name, output_commodity)
                ("electricity", "electrolyzer", "hydrogen"),
                ("electricity", "haber_bosch", "ammonia"),
                ("hydrogen", "haber_bosch", "ammonia")
            }

        **converter_upstreams** *(dict[tuple[str,str], list[str]])*: Keys are set of
        ``(input_commodity, tech_name)`` and the values are a set of
        upstream technologies that output the `input_commodity` to `tech_name`. An
        example of this variable is shown below.

            >>> converter_upstreams  # keys formatted as (input_commodity, tech)
            {
                #  (input_commodity, tech) : [techs that provide input_commodity to tech]
                ("electricity", "electrolyzer"): ["wind", "solar", "elec_combiner"],
                ("electricity", "haber_bosch"): ["electricity_feedstock"],
                ("hydrogen", "haber_bosch"): ["electrolyzer"],
                ("ammonia", "nh3_load_demand"): ["nh3_combiner", "nh3_storage", "haber_bosch"]
            }

    """
    in_flows = dict(technology_graph.in_degree)
    out_flows = dict(technology_graph.out_degree)

    non_converter_techs = [
        k for k in list(technology_graph.nodes) if in_flows[k] < 1 or out_flows[k] < 1
    ]
    likely_converter_techs = (
        set(technology_graph.nodes) - set(non_converter_techs) - set(storage_techs)
    ) & set(input_techs)

    converter_info = set()
    converter_upstreams = {}
    for converter in list(likely_converter_techs):
        # predecessors are upstream and directly connected to the converter
        predecessor_techs = set(technology_graph.predecessors(converter))
        # succesor techs are directly downstream of the converter
        successor_techs = set(technology_graph.successors(converter))

        input_commods = set()
        for upstream_tech in predecessor_techs:
            if (
                cmod := technology_graph.edges[upstream_tech, converter].get("commodity")
            ) is not None:
                if isinstance(cmod, str):
                    input_commods.add(cmod)
                else:
                    input_commods |= set(cmod)

        output_commods = set()
        for downstream_tech in list(successor_techs):
            if (
                cmod := technology_graph.edges[converter, downstream_tech].get("commodity")
            ) is not None:
                if isinstance(cmod, str):
                    output_commods.add(cmod)
                else:
                    output_commods |= set(cmod)
        # A converter has commodities that appear only on one side:
        # upstream-only commodities are consumed, output-only are produced.
        consumed = input_commods - output_commods
        produced = output_commods - input_commods
        if consumed and produced:
            # If both sides have unique commodities, this tech is a converter

            for input_commod in input_commods:
                upstream_techs_with_commod = get_successors_for_tech_with_input_cmod(
                    technology_graph, converter, input_commod
                )
                converter_upstreams[(input_commod, converter)] = upstream_techs_with_commod
                for output_commod in output_commods:
                    converter_info.add((input_commod, converter, output_commod))

    return converter_info, converter_upstreams
