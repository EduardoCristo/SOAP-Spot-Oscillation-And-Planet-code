import inspect
from collections import namedtuple
from functools import wraps

import numpy as np
from astropy import units as U
from astropy.units.core import UnitBase, UnitsError, add_enabled_equivalencies
from astropy.units.decorators import QuantityInput, _get_allowed_units
from astropy.utils.misc import isiterable

pp1 = U.Unit(1)
ppm = U.Unit(1e-6)
ms = U.meter / U.second
kms = U.kilometer / U.second
Rsun = U.R_sun


def has_unit(value):
    return isinstance(value, U.Quantity)


def unit_arange(start, stop, step):
    """A version of np.arange which works with Quantity arguments"""
    unit = start.unit
    r = np.arange(start.value, stop.to(unit).value, step.to(unit).value)
    return r * unit


def without_units(obj):
    """
    Create an object with the same attributes as `obj` but getting the .value of
    the attributes that are astropy Quantities. The new object can be more
    safely used when calling C functions.
    """
    name = obj.__class__.__name__
    objdir = obj.__dir__()
    attrs = [a for a in objdir if a[0] != "_" and not callable(getattr(obj, a))]
    callables = [a for a in objdir if a[0] != "_" and callable(getattr(obj, a))]

    new = namedtuple(name, attrs + callables)

    values = {}
    for attr in attrs:
        try:
            values[attr] = getattr(getattr(obj, attr), "value")
        except AttributeError:
            values[attr] = getattr(obj, attr)

    for call in callables:
        values[call] = getattr(obj, call)

    return new(**values)


# the following is mostly a copy from astropy.units.decorators, just patching
# the _validate_arg_value function to allow for unitless arguments


def _validate_arg_value(param_name, func_name, arg, targets, equivalencies):
    """
    Validates the object passed in to the wrapped function, ``arg``, with target
    unit or physical type, ``target``.
    """

    if len(targets) == 0:
        return

    allowed_units = _get_allowed_units(targets)
    if not hasattr(arg, "unit"):
        return arg * allowed_units[0]

    for allowed_unit in allowed_units:
        try:
            is_equivalent = arg.unit.is_equivalent(
                allowed_unit, equivalencies=equivalencies
            )

            if is_equivalent:
                break

        except AttributeError:  # Either there is no .unit or no .is_equivalent
            if hasattr(arg, "unit"):
                error_msg = "a 'unit' attribute without an 'is_equivalent' method"
                raise TypeError(
                    "Argument '{}' to function '{}' has {}. "
                    "You may want to pass in an astropy Quantity instead.".format(
                        param_name, func_name, error_msg
                    )
                )
            else:
                # we're good!
                break

    else:
        if len(targets) > 1:
            raise UnitsError(
                "Argument '{}' to function '{}' must be in units"
                " convertible to one of: {}.".format(
                    param_name, func_name, [str(targ) for targ in targets]
                )
            )
        else:
            raise UnitsError(
                "Argument '{}' to function '{}' must be in units"
                " convertible to '{}'.".format(param_name, func_name, str(targets[0]))
            )


class MaybeQuantityInput(QuantityInput):

    def __call__(self, wrapped_function):

        # Extract the function signature for the function we are wrapping.
        wrapped_signature = inspect.signature(wrapped_function)

        # Define a new function to return in place of the wrapped one
        @wraps(wrapped_function)
        def wrapper(*func_args, **func_kwargs):
            # Convert from tuple to list so it's modifiable
            func_args = list(func_args)

            # Bind the arguments to our new function to the signature of the original.
            bound_args = wrapped_signature.bind(*func_args, **func_kwargs)

            # Iterate through the parameters of the original signature
            for i, param in enumerate(wrapped_signature.parameters.values()):
                # We do not support variable arguments (*args, **kwargs)
                if param.kind in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue

                # Catch the (never triggered) case where bind relied on a default value.
                if (
                    param.name not in bound_args.arguments
                    and param.default is not param.empty
                ):
                    bound_args.arguments[param.name] = param.default

                # Get the value of this parameter (argument to new function)
                arg = bound_args.arguments[param.name]

                # Get target unit or physical type, either from decorator kwargs
                #   or annotations
                if param.name in self.decorator_kwargs:
                    targets = self.decorator_kwargs[param.name]
                    is_annotation = False
                else:
                    targets = param.annotation
                    is_annotation = True

                # If the targets is empty, then no target units or physical
                #   types were specified so we can continue to the next arg
                if targets is inspect.Parameter.empty:
                    continue

                # If the argument value is None, and the default value is None,
                #   pass through the None even if there is a target unit
                if arg is None and param.default is None:
                    continue

                # Here, we check whether multiple target unit/physical type's
                #   were specified in the decorator/annotation, or whether a
                #   single string (unit or physical type) or a Unit object was
                #   specified
                if isinstance(targets, str) or not isiterable(targets):
                    valid_targets = [targets]

                # Check for None in the supplied list of allowed units and, if
                #   present and the passed value is also None, ignore.
                elif None in targets:
                    if arg is None:
                        continue
                    else:
                        valid_targets = [t for t in targets if t is not None]

                else:
                    valid_targets = targets

                # If we're dealing with an annotation, skip all the targets that
                #    are not strings or subclasses of Unit. This is to allow
                #    non unit related annotations to pass through
                if is_annotation:
                    valid_targets = [
                        t for t in valid_targets if isinstance(t, (str, UnitBase))
                    ]

                # Now we loop over the allowed units/physical types and validate
                #   the value of the argument. If the argument does not have a
                #   `.unit`, it's returned having the first unit in the list of
                #   valid_targets
                arg = _validate_arg_value(
                    param.name,
                    wrapped_function.__name__,
                    arg,
                    valid_targets,
                    self.equivalencies,
                )

                if arg is not None:
                    try:
                        func_args[i] = arg
                    except IndexError:
                        func_kwargs[param.name] = arg

            # Call the original function with any equivalencies in force.
            with add_enabled_equivalencies(self.equivalencies):
                return_ = wrapped_function(*func_args, **func_kwargs)
            if wrapped_signature.return_annotation not in (
                inspect.Signature.empty,
                None,
            ):
                return return_.to(wrapped_signature.return_annotation)
            else:
                return return_

        return wrapper


maybe_quantity_input = MaybeQuantityInput.as_decorator
