"""
Utilities to work with Python's flawed but useful type system.
"""

import collections
from typing import List, Type, Generic, Union


def get_contained_type(expected_type: Type) -> List[Type]:
    """
    Try to get the types of the members of a List, Tuple or Union.
    WARNING: This function uses unstable APIs, and might break at some point in the future.
    """
    from typing import _GenericAlias, T  # ignore: typing
    if isinstance(expected_type, _GenericAlias):
        res = list(expected_type.__args__)
        if get_origin(expected_type) is collections.abc.Callable and res[0] is not Ellipsis:
            res = [list(res[:-1]), res[-1]]
        if len(res) == 1 and res[
            0] is T:  # check for passing in things like List (without any internal types)
            return []
        return res
    else:
        return []


def get_origin(tp: Type):
    """
    For container types, get_origin returns the container type.
    :param tp:
    :return:
    """
    from typing import _GenericAlias  # ignore: typing
    if isinstance(tp, _GenericAlias):
        return tp.__origin__
    if tp is Generic:
        return Generic
    return None


def is_union_type(tp: Type) -> bool:
    """
    Is @expected_type a Union (e.g. Optional[X])?
    """
    from typing import _GenericAlias  # ignore: typing
    return (tp is Union or
            isinstance(tp, _GenericAlias) and tp.__origin__ is Union)


def canonical_type_name(klass: Type) -> str:
    if klass.__module__ is not None:
        return "{}.{}".format(klass.__module__, klass.__name__)
    else:
        return klass.__name__


__all__ = ['get_contained_type', 'is_union_type', 'canonical_type_name', 'get_origin']
