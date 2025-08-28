from h2integrate.core.supported_models import commodity_to_model_kwargs


def make_model_kwarg_to_commodity():
    model_kwargs_to_commodity = {}
    for commodity, tech_kwargs in commodity_to_model_kwargs.items():
        model_kwargs_to_commodity.update({tech_kw: commodity for tech_kw in tech_kwargs})
    return model_kwargs_to_commodity


def get_commodities_for_tech_keywords(
    finance_group_tech_config: dict, sub_dict=None, sub_dict_key=None
):
    model_kwargs_to_commodity = make_model_kwarg_to_commodity()
    commodity_types = set()
    values_to_check = []
    if sub_dict is None and sub_dict_key is None:
        values_to_check = list(finance_group_tech_config.keys())
    if sub_dict is not None and sub_dict_key is not None:
        for tech_dict in finance_group_tech_config.values():
            val = tech_dict.get(sub_dict, {}).get(sub_dict_key, None)
            if val is not None and isinstance(val, str):
                values_to_check.append(val)

    for value in values_to_check:
        commodity = [v for k, v in model_kwargs_to_commodity.items() if value.lower() in k]
        if len(commodity) > 0:
            # for commod in commodity:
            commodity_types.update(commodity)
        # else:
        #     msg = (
        #         f"Technology {tech_name} name has unrecognized commodity type."
        #         "Please check `commodity_to_model_kwargs` in "
        #         "h2integrate/core/supported_models.py to find valid technology keywords."
        #     )
        #     raise ValueError(msg)
    # unique_commodity_types = list(set(commodity_types))
    return list(commodity_types)


def get_commodity_for_technology_name(tech_name, tech_params, check_model_name: str | None = None):
    if check_model_name is not None and check_model_name.lower() not in [
        "performance_model",
        "cost_model",
        "finance_model",
    ]:
        msg = "Check model name must be either 'performance_model','cost_model', or 'finance_model'"
        raise ValueError(msg)
    model_kwargs_to_commodity = make_model_kwarg_to_commodity()
    commodities = set()
    commodity_from_name = [
        v for k, v in model_kwargs_to_commodity.items() if tech_name.lower() in k
    ]
    commodities.update(commodity_from_name)
    if check_model_name is not None:
        model_name = tech_params.get(check_model_name, {}).get("model", None)
        if model_name is not None:
            commodity_from_model_name = [
                v for k, v in model_kwargs_to_commodity.items() if model_name.lower() in k
            ]
            commodities.update(commodity_from_model_name)
    return list(commodities)
