"""
An extensible typed configuration class.
"""
import json
import typing
from enum import Enum
from numbers import Real, Integral
from typing import get_type_hints, Any, List, Type, TypeVar, Dict

from .type_inspection import get_contained_type, is_union_type, get_origin

T = TypeVar('T', bound='ConfigDict')
X = TypeVar('X')


def _type_check(value: Any, expected_type: Type) -> bool:
    """
    :param value: A value to type check.
    :param expected_type: A typing specification. It can be a Generic type like List[X] or Optional[Y].
    :return: true iff @value conforms to the type specification of @expected_type
    :raise: ValueError if expected_type is not a supported type.
    """
    # special case floats (sigh)
    if expected_type is type(None):
        return value is None
    if expected_type is bool:
        return isinstance(value, bool)
    elif expected_type is float:
        return isinstance(value, Real)
    elif expected_type is int:
        return isinstance(value, Integral)
    elif expected_type is str:
        return isinstance(value, str)
    elif expected_type is dict:
        return isinstance(value, dict)
    elif isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return isinstance(value, str) or isinstance(value, expected_type)
    elif isinstance(expected_type, type) and issubclass(expected_type, ConfigDict):
        if not isinstance(value, dict):
            # expects a dict version of this value
            return False
        try:
            expected_type(**value)
            return True
        except ValueError:
            return False
    elif get_origin(expected_type) in [list, List]:
        if not isinstance(value, List):
            return False
        sub_type, = get_contained_type(expected_type)
        return all(_type_check(elem, sub_type) for elem in value)
    elif get_origin(expected_type) in [dict, Dict]:
        if not isinstance(value, Dict):
            return False
        key_type, val_type = get_contained_type(expected_type)
        return all(_type_check(elem, key_type) for elem in value.keys()) \
               and all(_type_check(elem, val_type) for elem in value.values())
    elif is_union_type(expected_type):
        return any(_type_check(value, sub_type) for sub_type in get_contained_type(expected_type))
    else:
        raise ValueError(f"Unable to test against unsupported type: {expected_type}")


def _cast(value: Any, expected_type: Type[X]) -> X:
    """
    :param value: The value to cast
    :param expected_type: The type it should cast to
    :return: Type of X
    """
    if expected_type in [bool, float, int, str, dict, type(None)]:
        return value
    elif isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return typing.cast(X, expected_type[value])
    elif isinstance(expected_type, type) and issubclass(expected_type, ConfigDict):
        if not isinstance(value, dict):
            raise ValueError(f"Expects a dict version of {expected_type}, but got a {type(value)}")
        return typing.cast(X, expected_type(**value))
    elif get_origin(expected_type) in [list, List]:
        sub_type, = get_contained_type(expected_type)
        return typing.cast(X, [_cast(elem, sub_type) for elem in value])
    elif get_origin(expected_type) in [dict, Dict]:
        key_type, val_type = get_contained_type(expected_type)
        return typing.cast(X, {_cast(k, key_type): _cast(v, val_type) for k, v in value.items()})
    elif is_union_type(expected_type):
        # find the type that we should cast to
        for possible_type in get_contained_type(expected_type):
            if _type_check(value, possible_type):
                # noinspection PyTypeChecker
                return _cast(value, possible_type)
        raise ValueError(f"Unable to cast to unsupported type: {expected_type}")
    else:
        raise ValueError(f"Unable to cast to unsupported type: {expected_type}")


def _type_check_and_cast(value: Any, expected_type: Type[X]) -> X:
    """
    :param value: A value to type check and cast.
    :param expected_type: A typing specification. It can be a Generic type like List[X] or Optional[Y].
    :return: value cast to X. This particular version handles casts into ConfigDicts.
    :raise: ValueError if expected_type is not a supported type or the type cast failed.
    """
    _type_error = ValueError(f"Expects a {expected_type}, but got a {type(value)}")

    if _type_check(value, expected_type):
        return _cast(value, expected_type)
    else:
        raise _type_error


class ConfigDict:
    @property
    def _fields(self):
        """
        Get the typed public fields of this config dict.
        :return: A list of public fields which have types.
        """
        types = get_type_hints(type(self))
        return [field for field in dir(self) if not field.startswith("_") and field in types]

    def __init__(self, **kwargs):
        self.__extra = {}  # Extra arguments that weren't processed here.
        self.update(**kwargs)

    def update(self, **kwargs):
        """Set fields in their class order"""
        set_fields = self._fields
        types = get_type_hints(type(self))

        # Check all required properties exist.
        for k in types:
            if k not in set_fields and k not in kwargs:
                raise ValueError(f"{k} is a required property")

        # For all specified properties, check their types
        # NOTE: we don't check the types of default values because we expect the type checker to handle these cases.
        for k in list(kwargs):
            # Skip anything that's not in our set of fields.
            if k not in types:
                continue

            v = kwargs.pop(k)
            expected_type: Type = types.get(k)

            try:
                # If self.{k} is a ConfigDict, we should recurse the 'update' call.
                if isinstance(expected_type, type) and issubclass(expected_type,
                                                                  ConfigDict) and hasattr(self, k):
                    if not isinstance(v, dict):
                        raise ValueError(f"Expects a {expected_type}, but got a {type(v)}")
                    getattr(self, k).update(**v)
                else:
                    # In all other cases, we override the field.
                    v = _type_check_and_cast(v, expected_type)
                    setattr(self, k, v)
            except ValueError as e:
                raise ValueError(f"{k} expects a {types.get(k)}, but got a {type(v)}.")

        # Validate entries
        try:
            self._validate()
        except AssertionError as e:
            raise ValueError(
                f"Provided entries did not validate: {e.args[0] if e.args else '(unspecified)'}")

        # Any unresolved arguments are stored in __extra to support upcasting.
        self.__extra.update(kwargs)

    def as_json(self) -> dict:
        """
        Converts a ConfigDict into a dictionary that can be serialized to/from JSON.
        :return:
        """

        ret = dict(self.__extra)
        for key in self._fields:
            value = ConfigDict._to_json_serializable(getattr(self, key))
            ret[key] = value

        return ret

    @staticmethod
    def _to_json_serializable(value: Any):
        if isinstance(value, ConfigDict):
            return value.as_json()
        elif isinstance(value, Enum):
            return value.name
        elif isinstance(value, List):
            return [ConfigDict._to_json_serializable(v) for v in value]
        elif isinstance(value, Dict):
            return {ConfigDict._to_json_serializable(k): ConfigDict._to_json_serializable(v)
                    for k, v in value.items()}
        else:
            return value

    @classmethod
    def cast(cls: Type[T], obj: 'ConfigDict') -> T:
        """
        Upcasts a config dict of type U into one of T.
        :param obj:
        :return:
        """
        if not issubclass(cls, obj.__class__):
            raise ValueError("Can only down-cast from a parent to a child class.")

        return typing.cast(T, cls(**obj.as_json()))

    def __eq__(self, other):
        return self.as_json() == other.as_json()

    def __hash__(self):
        # TODO Convert all lists to tuples.
        return super().__hash__()

    def __len__(self):
        return len(self._fields)

    def __iter__(self):
        return ((f, getattr(self, f)) for f in self._fields)

    def _validate(self):
        """
        Implementers can define a validate routine that ensures that the fields are valid (more than just a type check).
        :return:
        """
        pass

    def __str__(self):
        return json.dumps(self.as_json())


# region: config dict parser
try:
    # We use lark to parse configuration strings.
    import lark

    CONFIG_DICT_GRAMMAR = r"""
    start: expr*
    expr: key "=" _value
 
    key: CNAME ("." CNAME)*
    _value: none | int | float | string | _nested_expr | list_expr 
 
    none: "null"
    int: SIGNED_INT
    float: SIGNED_FLOAT
    string: WORD | ESCAPED_STRING
    list_expr: "[" [_value ("," _value)*] "]"
    _nested_expr: "(" start ")"
    
    WORD: /[.a-zA-Z0-9_\/-]+/
 
    %import common.SIGNED_INT       // imports from terminal library
    %import common.SIGNED_FLOAT     // imports from terminal library
    %import common.CNAME            // imports from terminal library
    %import common.ESCAPED_STRING   // imports from terminal library
    %import common.WS               // imports from terminal library
    %ignore WS                      // Disregard spaces in text
    """


    # noinspection PyMethodMayBeStatic PyMethodMayBeFunction
    def _update_dict(obj, key, value):
        if '.' in key:
            root_key, sub_key = key.split('.', 1)
            if root_key not in obj:
                obj[root_key] = {}
            _update_dict(obj[root_key], sub_key, value)
        else:
            obj[key] = value


    class _ConfigDictTransformer(lark.Transformer):
        def start(self, parts):
            ret = {}
            for key, value in parts:
                _update_dict(ret, key, value)

            return ret

        @lark.v_args(inline=True)
        def expr(self, k, v):
            return k, v

        def key(self, parts):
            return ".".join(parts)

        def list_expr(self, vs):
            return list(vs)

        @lark.v_args(inline=True)
        def none(self):
            return None

        @lark.v_args(inline=True)
        def int(self, s):
            return int(s)

        @lark.v_args(inline=True)
        def float(self, s):
            return float(s)

        @lark.v_args(inline=True)
        def string(self, s):
            # Special catch cases here
            if s.type == "WORD":
                if str(s.value).lower() == "true":
                    return True
                elif str(s.value).lower() == "false":
                    return False
                else:
                    return s.value
            else:
                assert s.type == "ESCAPED_STRING"
                return s.value[1:-1].replace('\\"', '"')


    ConfigDictParser = lark.Lark(CONFIG_DICT_GRAMMAR, parser='lalr',
                                 transformer=_ConfigDictTransformer())
except ImportError:
    lark = None


    class _ConfigDictParser:
        def parse(self, _):
            raise NotImplementedError("Can't parse input because lark-parser was not installed.")


    ConfigDictParser = _ConfigDictParser()
# endregion


ConfigDictParser.__doc__ = """
Parses a configuration from a string.
Here are some examples of usages that cover the features of this simple grammar:
 
- name="the \"best\" name" # parses to {'name': 'the "best" name'}  -- quote strings the usual way.
- epochs = 10  # parses to {'epochs': 10}                       -- note the transparent typing of numbers
- early_stop = true  # parses to {'epochs': True}               --
- optimizer.type=adam # parses to {'optimizer": {'type': 'adam'}}    -- by default we parse string expressions.
- ff_dims = [128, 128, 512] # parses to {"ff_dims": [128, 128, 512]} -- hierarchical expr.
- optimizer = (type=adam lr=1e-4 betas=[1e-4, 1e-5]) # parses to {"optimizer": {"type": "adam", "lr": 1e-4, "betas": [1e-4, 1e-5]}} -- hierarchical expr
 
For a full grammar, see CONFIG_DICT_GRAMMAR.
 
:param txt: text to parse.
:return:
"""

__all__ = ['ConfigDict', 'ConfigDictParser']
