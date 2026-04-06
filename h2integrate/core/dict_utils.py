import copy
import operator
from functools import reduce

import numpy as np


def dict_to_yaml_formatting(orig_dict):
    """Recursive method to convert arrays to lists and numerical entries to floats.
    This is primarily used before writing a dictionary to a YAML file to ensure
    proper output formatting.

    Args:
        orig_dict (dict): input dictionary

    Returns:
        dict: input dictionary with reformatted values.
    """
    for key, val in orig_dict.items():
        if isinstance(val, dict):
            tmp = dict_to_yaml_formatting(orig_dict.get(key, {}))
            orig_dict[key] = tmp
        else:
            if isinstance(key, list):
                for i, k in enumerate(key):
                    if isinstance(orig_dict[k], str | bool | int):
                        orig_dict[k] = orig_dict.get(k, []) + val[i]
                    elif isinstance(orig_dict[k], list | np.ndarray):
                        orig_dict[k] = np.array(val, dtype=float).tolist()
                    else:
                        orig_dict[k] = float(val[i])
            elif isinstance(key, str):
                if isinstance(orig_dict[key], str | bool | int):
                    continue
                if orig_dict[key] is None:
                    continue
                if isinstance(orig_dict[key], list | np.ndarray):
                    if any(isinstance(v, dict) for v in val):
                        for vii, v in enumerate(val):
                            if isinstance(v, dict):
                                new_val = dict_to_yaml_formatting(v)
                            else:
                                new_val = v if isinstance(v, str | bool | int) else float(v)
                            orig_dict[key][vii] = new_val
                    else:
                        new_val = [v if isinstance(v, str | bool | int) else float(v) for v in val]
                        orig_dict[key] = new_val
                else:
                    orig_dict[key] = float(val)
    return orig_dict


def remove_numpy(fst_vt: dict) -> dict:
    """
    Recursively converts numpy array elements within a nested dictionary to lists and ensures
    all values are simple types (float, int, dict, bool, str) for writing to a YAML file.

    Args:
        fst_vt (dict): The dictionary to process.

    Returns:
        dict: The processed dictionary with numpy arrays converted to lists
            and unsupported types to simple types.
    """

    def get_dict(vartree, branch):
        return reduce(operator.getitem, branch, vartree)

    # Define conversion dictionary for numpy types
    conversions = {
        np.int_: int,
        np.intc: int,
        np.intp: int,
        np.int8: int,
        np.int16: int,
        np.int32: int,
        np.int64: int,
        np.uint8: int,
        np.uint16: int,
        np.uint32: int,
        np.uint64: int,
        np.single: float,
        np.double: float,
        np.longdouble: float,
        np.csingle: float,
        np.cdouble: float,
        np.float16: float,
        np.float32: float,
        np.float64: float,
        np.complex64: float,
        np.complex128: float,
        np.bool_: bool,
        np.ndarray: lambda x: x.tolist(),
    }

    def loop_dict(vartree, branch):
        if not isinstance(vartree, dict):
            return fst_vt
        for var in vartree.keys():
            branch_i = copy.copy(branch)
            branch_i.append(var)
            if isinstance(vartree[var], dict):
                loop_dict(vartree[var], branch_i)
            else:
                current_value = get_dict(fst_vt, branch_i[:-1])[branch_i[-1]]
                data_type = type(current_value)
                if data_type in conversions:
                    get_dict(fst_vt, branch_i[:-1])[branch_i[-1]] = conversions[data_type](
                        current_value
                    )
                elif isinstance(current_value, list | tuple):
                    for i, item in enumerate(current_value):
                        current_value[i] = remove_numpy(item)

    # set fast variables to update values
    loop_dict(fst_vt, [])
    return fst_vt


def update_defaults(orig_dict, keyname, new_val):
    """Recursive method to update all entries in a dictionary with key 'keyname'
    with value 'new_val'

    Args:
        orig_dict (dict): dictionary to update
        keyname (str): key corresponding to value to update
        new_val (any): value to use for ``keyname``

    Returns:
        dict: updated version of orig_dict
    """
    for key, val in orig_dict.items():
        if isinstance(val, dict):
            tmp = update_defaults(orig_dict.get(key, {}), keyname, new_val)
            orig_dict[key] = tmp
        else:
            if isinstance(key, list):
                for i, k in enumerate(key):
                    if k == keyname:
                        orig_dict[k] = new_val
                    else:
                        orig_dict[k] = orig_dict.get(key, []) + val[i]
            elif isinstance(key, str):
                if key == keyname:
                    orig_dict[key] = new_val
    return orig_dict


def update_keyname(orig_dict, init_key, new_keyname):
    """Recursive method to copy value of ``orig_dict[init_key]`` to ``orig_dict[new_keyname]``

    Args:
        orig_dict (dict): dictionary to update.
        init_key (str): existing key
        new_keyname (str): new key to replace ``init_key``

    Returns:
        dict: updated dictionary
    """

    for key, val in orig_dict.copy().items():
        if isinstance(val, dict):
            tmp = update_keyname(orig_dict.get(key, {}), init_key, new_keyname)
            orig_dict[key] = tmp
        else:
            if isinstance(key, list):
                for i, k in enumerate(key):
                    if k == init_key:
                        orig_dict.update({new_keyname: orig_dict.get(k)})
                    else:
                        orig_dict[k] = orig_dict.get(key, []) + val[i]
            elif isinstance(key, str):
                if key == init_key:
                    orig_dict.update({new_keyname: orig_dict.get(key)})
    return orig_dict


def remove_keynames(orig_dict, init_key):
    """Recursive method to remove keys from a dictionary.

    Args:
        orig_dict (dict): input dictionary
        init_key (str): key name to remove from dictionary

    Returns:
        dict: dictionary without any keys named `init_key`
    """

    for key, val in orig_dict.copy().items():
        if isinstance(val, dict):
            tmp = remove_keynames(orig_dict.get(key, {}), init_key)
            orig_dict[key] = tmp
        else:
            if isinstance(key, list):
                for i, k in enumerate(key):
                    if k == init_key:
                        orig_dict.pop(k)
                    else:
                        orig_dict[k] = orig_dict.get(key, []) + val[i]
            elif isinstance(key, str):
                if key == init_key:
                    orig_dict.pop(key)
    return orig_dict


def rename_dict_keys(input_dict, init_keyname, new_keyname):
    """Rename ``input_dict[init_keyname]`` to ``input_dict[new_keyname]``

    Args:
        input_dict (dict): dictionary to update
        init_keyname (str): existing key to replace
        new_keyname (str): new keyname

    Returns:
        dict: updated dictionary
    """
    input_dict = update_keyname(input_dict, init_keyname, new_keyname)
    input_dict = remove_keynames(input_dict, init_keyname)
    return input_dict


def check_inputs(prob, tech: str, tech_info: dict):
    """Check the user-input technology configuration inputs against the
    instantiated technology configuration classes to ensure that:

    1. All user-input parameters are used in at least 1 configuration class
    2. User-input `shared_parameters` are shared across at least 2 configuration classes
    3. User-input parameters that are not-shared are only used in 1 configuration class

    Args:
        prob (om.Problem): OpenMDAO problem defined in H2IntegrateModel
        tech (str): name of technology that the tech_info is for.
        tech_info (dict): technology input dictionary, including the
            technology model names and `model_inputs`.

    Raises:
        AttributeError: _description_
    """
    # Only check models that have a control strategy or dispatch rule set
    if not {"control_strategy", "dispatch_rule_set"}.intersection(tech_info):
        return

    # Get the technology group from the plant model
    group = getattr(prob.model.plant, tech)

    # Only check for shared inputs when the system contains at least one technology
    # in addition to a performance and control model
    check_keys = ("control_strategy", "dispatch_rule_set", "cost_model", "performance_model")
    minimal_keys = {"control_strategy", "performance_model"}
    overlap = set(tech_info).intersection(check_keys)
    if overlap == minimal_keys or len(overlap) < 3:
        return

    msg = None
    control_sys = None
    dispatch_sys = None
    cost_sys = None
    perf_sys = None

    # Rebuild the model inputs dictionary from the initialized technology parameters
    restructured_params = {}
    if "control_strategy" in tech_info:
        if (control_sys := getattr(group, tech_info["control_strategy"]["model"])) is not None:
            restructured_params["control_parameters"] = control_sys.config.as_dict()
    if "dispatch_rule_set" in tech_info:
        if (dispatch_sys := getattr(group, tech_info["dispatch_rule_set"]["model"])) is not None:
            restructured_params["dispatch_parameters"] = dispatch_sys.config.as_dict()
    if "cost_model" in tech_info:
        if (cost_sys := getattr(group, tech_info["cost_model"]["model"])) is not None:
            restructured_params["cost_parameters"] = cost_sys.config.as_dict()
    if "performance_model" in tech_info:
        if (perf_sys := getattr(group, tech_info["performance_model"]["model"])) is not None:
            restructured_params["performance_parameters"] = perf_sys.config.as_dict()

    # Reconstruct the shared_parameters part of model_inputs
    shared_params = {}
    for param_key, v in restructured_params.items():
        other_keys = [ok for ok in restructured_params.keys() if ok != param_key]
        for other_key in other_keys:
            if any(ok in v for ok in restructured_params[other_key].keys()):
                # get keys shared between other_key and param_key
                shared_other_param = {
                    ok: ov for ok, ov in restructured_params[other_key].items() if ok in v
                }
                shared_params.update(shared_other_param)
                # remove the shared params from other_key dictionary
                other_key_items = {
                    k: v
                    for k, v in restructured_params[other_key].items()
                    if k not in shared_params
                }
                restructured_params[other_key] = other_key_items
        # remove shared params from param_key
        param_key_items = {
            k: v for k, v in restructured_params[param_key].items() if k not in shared_params
        }
        restructured_params[param_key] = param_key_items

    restructured_params["shared_parameters"] = shared_params
    # Now, restructured_params is what the model_inputs configuration should be

    # Check each parameter dictionary of the restructured model_inputs against the user-provided
    # model_inputs by looping through the parameters in the restructed_parameters dictionary.
    # `param_key` is 'performance_parameters', 'control_parameters', 'shared_parameters', etc
    for param_key in restructured_params.keys():
        # check that the parameter key exists in both the user-provided model_inputs and
        # the restructured parameters
        if param_key in tech_info["model_inputs"] and param_key in restructured_params:
            # Get the difference between the user-input parameters and the restructured
            # parameters
            dict_differences = {
                k: tech_info["model_inputs"][param_key][k]
                for k in set(tech_info["model_inputs"][param_key])
                - set(restructured_params[param_key])
            }
            # Only check for keys that are defined in the user-input parameters that aren't
            # found in the restructured parameters to avoid throwing errors when users did
            # not provide optional configuration inputs
            if len(dict_differences) > 0:
                if param_key == "shared_parameters":
                    # check if the parameter is not shared, but used by one tech
                    for other_key, other_params in restructured_params.items():
                        if other_key == param_key:
                            continue
                        if any(k in other_params for k in dict_differences):
                            unshared_params = [k for k in dict_differences if k in other_params]
                            msg = (
                                f"The parameter(s): {unshared_params} found in "
                                f"shared_parameters but should be in {other_key} "
                                f"for technology {tech}"
                            )
                            raise AttributeError(msg)

                    if msg is None:
                        # the parameter is not used by any tech
                        msg = (
                            f"The parameter(s): {list(dict_differences.keys())} found in "
                            f"shared_parameters are not used by any of the models for "
                            f"technology {tech}"
                        )
                        raise AttributeError(msg)

                else:
                    # check if parameter is shared but only under one technology
                    if any(
                        k in restructured_params.get("shared_parameters", {})
                        for k in dict_differences
                    ):
                        should_be_shared_keys = [
                            k
                            for k in dict_differences
                            if k in restructured_params.get("shared_parameters", {})
                        ]
                        msg = (
                            f"The parameter(s) {should_be_shared_keys} found in "
                            f"{param_key} should be under shared_parameter(s) for "
                            f"technology {tech}"
                        )
                        raise AttributeError(msg)

                    msg = (
                        f"The parameter(s) {list(dict_differences.keys())} found in "
                        f"{param_key} are not used for technology {tech}"
                    )
                    raise AttributeError(msg)

    if msg is None:
        return
