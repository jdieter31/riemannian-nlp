import enum
import json
import unittest
from typing import List, Optional, Dict

from pynlp.utils.config import ConfigDict, _type_check, ConfigDictParser
from pynlp.utils.type_inspection import get_contained_type


class ConfigA(ConfigDict):
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    field_a: float = 0.4
    field_b: str = "Test A"


class ConfigB(ConfigA):
    required_a: int
    field_b: str = "Test AB"
    field_c: float = 0.5
    field_d: str = "Test B"


class ConfigC(ConfigDict):
    a: ConfigA = ConfigA()
    field_c: float = 0.5
    field_d: str = "Test B"


class ConfigD(ConfigDict):
    a_list: List[ConfigA] = [ConfigA()]
    a_dict: Dict[str, ConfigA] = {"a_key": ConfigA()}


class Base(ConfigDict):
    field_a: float = 1e-2


class VariantA(Base):
    field_a: float = 1e-3
    special: str = "apple"


class VariantB(Base):
    field_a: float = 1e-2
    special: str = "oranges"
    is_citrus: bool = True


class VariantC(Base):
    field_a: float = 1e-1
    special: int = 3
    bounds: List[int] = [1, 2]


class AnEnum(enum.Enum):
    APPLE = 1
    BANANA = 2
    CARROT = "3"


class WithEnum(ConfigDict):
    field: AnEnum = AnEnum.APPLE


class ConfigDictTest(unittest.TestCase):

    def test_generic_types(self):
        """
        Test that _get_generic_types works : this is an unstable interface, so all the more reason to test.
        :return:
        """
        self.assertEqual([int, ], get_contained_type(List[int]))
        self.assertEqual([ConfigA, ], get_contained_type(List[ConfigA]))
        self.assertEqual([int, type(None), ], get_contained_type(Optional[int]))

    def test_type_checking(self):
        self.assertTrue(_type_check(False, bool))
        self.assertFalse(_type_check(0, bool))

        self.assertTrue(_type_check(3, float))
        self.assertTrue(_type_check(3.0, float))
        self.assertFalse(_type_check("3", float))

        self.assertTrue(_type_check(3, int))
        self.assertFalse(_type_check(3.0, int))
        self.assertFalse(_type_check("3", int))

        self.assertTrue(_type_check("3", str))
        self.assertFalse(_type_check(3, str))

        self.assertTrue(_type_check([], List[int]))
        self.assertTrue(_type_check([3, 4], List[int]))
        self.assertFalse(_type_check("4", List[int]))
        self.assertFalse(_type_check([3, "4"], List[int]))

        self.assertTrue(_type_check(3, Optional[int]))
        self.assertTrue(_type_check(None, Optional[int]))
        self.assertFalse(_type_check("4", Optional[int]))

        # test type check None
        self.assertTrue(_type_check(None, type(None)))
        self.assertFalse(_type_check(3, type(None)))

        # test type check totally unsupported type
        class Unsupported:
            pass

        with self.assertRaises(ValueError):
            self.assertFalse(_type_check(3, Unsupported))

        # test config dict type but pass in non dict
        self.assertFalse(_type_check(3, ConfigA))

        # test dict type but pass in non dict
        self.assertFalse(_type_check(3, dict))

        # test type check with error inside nested config dict
        class BadValidate(ConfigDict):
            a: int

            def _validate(self):
                assert False

        self.assertFalse(_type_check({'a': 3}, BadValidate))

    def test_default(self):
        config = ConfigA()
        self.assertAlmostEqual(config.field_a, 0.4)
        self.assertEqual(config.field_b, "Test A")

    def test_setting(self):
        config = ConfigA(field_b="TestTest A", field_a=4)
        self.assertAlmostEqual(config.field_a, 4.)
        self.assertEqual(config.field_b, "TestTest A")

    def test_typing(self):
        self.assertRaises(ValueError, lambda: ConfigA(field_b=3.14))

    def test_subclassing(self):
        config = ConfigB(required_a=3)
        self.assertAlmostEqual(config.field_a, 0.4)
        self.assertEqual(config.field_b, "Test AB")
        self.assertAlmostEqual(config.field_c, 0.5)
        self.assertEqual(config.field_d, "Test B")
        self.assertEqual(config.required_a, 3)

        self.assertRaises(ValueError, lambda: ConfigB(required_a=3, field_b=3.14))

    def test_required_properties(self):
        self.assertRaises(ValueError, lambda: ConfigB())

    def test_serialization(self):
        config = ConfigB(required_a=4)
        self.assertEqual(config.as_json(), {
            "field_a": 0.4,
            "field_b": "Test AB",
            "field_c": 0.5,
            "field_d": "Test B",
            "required_a": 4,
        })
        config_ = ConfigB(**config.as_json())
        self.assertEqual(config, config_)

    def test_nested_serialization(self):
        config = ConfigC()
        self.assertEqual(config.as_json(), {
            "a": {
                "field_a": 0.4,
                "field_b": "Test A",
            },
            "field_c": 0.5,
            "field_d": "Test B",
        })
        config_ = ConfigC(**json.loads(json.dumps(config.as_json())))
        self.assertEqual(config, config_)

        config = ConfigD()
        config_ = ConfigD(**json.loads(json.dumps(config.as_json())))
        self.assertEqual(config, config_)

    def test_casting(self):
        # Without values.
        config = Base()
        self.assertEqual(config.as_json(), {
            "field_a": 1e-2,
        })

        self.assertEqual({
            # Note that this value is not going to be updated because up-casting can't override default values.
            "field_a": 1e-2,
            "special": "apple",
        }, VariantA.cast(config).as_json())

        self.assertEqual({
            "field_a": 1e-2,
            "special": "oranges",
            "is_citrus": True,
        }, VariantB.cast(config).as_json())

        self.assertEqual({
            "field_a": 1e-2,
            "special": 3,
            "bounds": [1, 2],
        }, VariantC.cast(config).as_json())

        data = {
            "special": 5,
            "bounds": [4, 6],
        }

        config = Base(**data)
        self.assertEqual({
            "field_a": 1e-2,
            "special": 5,
            "bounds": [4, 6],
        }, VariantC.cast(config).as_json())

        class Unsupported:
            pass

        class BadConfig(ConfigDict):
            field_a: Unsupported

        config = Base(**{'field_a': 1e-2})
        with self.assertRaises(ValueError):
            BadConfig.cast(config)

    def test_enums(self):
        self.assertEqual({"field": "APPLE"}, WithEnum().as_json())
        self.assertEqual(AnEnum.APPLE, WithEnum(field="APPLE").field)
        self.assertEqual(AnEnum.BANANA, WithEnum(field="BANANA").field)
        self.assertEqual(AnEnum.CARROT, WithEnum(field="CARROT").field)

    def test_parsing(self):
        self.assertEqual({}, ConfigDictParser.parse(''))

        # String quoting
        self.assertEqual({'name': 'the "best" name'},
                         ConfigDictParser.parse('name="the \\"best\\" name"'))
        self.assertEqual({'classes': ["a.b", "b.c", "c.d", "none"]},
                         ConfigDictParser.parse('classes = [a.b, b.c, c.d, "none"]'))

        # Typing
        self.assertEqual({'epochs': 10}, ConfigDictParser.parse('epochs=10'))
        self.assertEqual({'early_stop': True}, ConfigDictParser.parse('early_stop=true'))

        # Lists
        self.assertEqual({'ff_dim': [128, 128, 512]},
                         ConfigDictParser.parse('ff_dim=[128, 128, 512]'))
        self.assertEqual({'classes': ["general.yes", "general.no"]}, ConfigDictParser
                         .parse('classes=[\"general.yes\",\"general.no\"]'))

        # Hierarchical 1
        self.assertEqual({'optimizer': {"type": "adam"}},
                         ConfigDictParser.parse('optimizer.type=adam'))

        # Hierarchical 2
        self.assertEqual({'optimizer': {
            "type": "adam", "lr": 1e-4, "betas": [1e-4, 1e-5],
            "scheduler": {"type": "ReduceLROnPlateau"}}
        }, ConfigDictParser.parse(
            'optimizer=(type=adam lr=1e-4 betas=[1e-4, 1e-5] scheduler=(type=ReduceLROnPlateau))')
        )

        # Dicts
        self.assertEqual({'a': {'hi': 1, 'hello': 2}}, ConfigDictParser.parse("a.hi=1 a.hello=2"))

        # Booleans
        self.assertEqual({'a': True, 'b': False},
                         ConfigDictParser.parse("a=true b=false"))
        self.assertEqual({'a': True, 'b': False},
                         ConfigDictParser.parse("a=True b=False"))

    def test_nested_update(self):
        class A(ConfigDict):
            field1: str
            field2: str

        class B(ConfigDict):
            a: A

        cfg = B(**ConfigDictParser.parse("a.field1=x a.field2=y"))
        self.assertEqual({"a": {"field1": "x", "field2": "y"}}, cfg.as_json())
        cfg.update(**ConfigDictParser.parse("a.field1=z"))
        self.assertEqual({"a": {"field1": "z", "field2": "y"}}, cfg.as_json())

    def test_validate(self):
        class A(ConfigDict):
            field1: str
            field2: str

            def _validate(self):
                assert self.field1 == "correct"

        self.assertEqual(
            {'field1': 'correct', 'field2': 'something'},
            A(**{'field1': 'correct', 'field2': 'something'}).as_json()
        )
        with self.assertRaises(ValueError):
            A(**{'field1': 'wrong', 'field2': 'something'})

    def test_nested_configs(self):
        class Parent:
            class A(ConfigDict):
                field1: str

        class B(Parent.A):
            pass

        class D(B):
            field2: str = "b"

        with self.assertRaises(ValueError):
            D()

        cfg = D(field1="a")
        self.assertEqual({"field1": "a", "field2": "b"}, cfg.as_json())
        self.assertEqual("a", cfg.field1)
        self.assertEqual("b", cfg.field2)

    def test_optional_subconfig(self):
        class OptionalSubConfig(ConfigDict):
            field: Optional[ConfigA] = None

        cfg = OptionalSubConfig()
        self.assertEqual({"field": None}, cfg.as_json())
        cfg = OptionalSubConfig(field={"field_a": 1.2, "field_b": "b"})
        self.assertEqual({"field": {"field_a": 1.2, "field_b": "b"}}, cfg.as_json())
        self.assertIsInstance(cfg.field, ConfigA)

    def test_list(self):
        class A(ConfigDict):
            field1: List[float]

        cfg = A(field1=[1.1, 1.2, 1.3])
        self.assertEqual({"field1": [1.1, 1.2, 1.3]}, cfg.as_json())
        self.assertEqual(1.1, cfg.field1[0])
        self.assertEqual(1.2, cfg.field1[1])
        self.assertEqual(1.3, cfg.field1[2])

        with self.assertRaises(ValueError):
            A(field1=["1.1", "1.2", "1.3"])

    def test_dict(self):
        class A(ConfigDict):
            field1: Dict[str, int]

        cfg = A(field1={"a": 1, "b": 2})
        self.assertEqual({"field1": {"a": 1, "b": 2}}, cfg.as_json())
        self.assertEqual(1, cfg.field1["a"])
        self.assertEqual(2, cfg.field1["b"])

        with self.assertRaises(ValueError):
            A(field1={"a": 1, "b": "b"})
