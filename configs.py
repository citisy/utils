"""utils for configs, usually used in project startup, function definition, etc."""
import copy
import subprocess
from types import NoneType
from typing import List, Union, get_origin, get_args
from . import os_lib, converter


class ArgDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax.
    so that it can be treated as `argparse.ArgumentParser().parse_args()`"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class ConfigObjParse:
    convert_to_constant_value = False

    @staticmethod
    def collapse_dict(d: dict):
        """

        Example:
            >>> d = {'a': {'b': 1, 'c': 2, 'e': {'f': 4}}, 'd': 3}
            >>> ConfigObjParse.collapse_dict(d)
            >>> {'a.b': 1, 'a.c': 2, 'a.e.f': 4, 'd': 3}

        """

        def cur(cur_dic, cur_k, new_dic):
            for k, v in cur_dic.items():
                if isinstance(v, dict):
                    k = f'{cur_k}.{k}'
                    cur(v, k, new_dic)
                else:
                    new_dic[f'{cur_k}.{k}'] = v

            return new_dic

        new_dic = cur(d, '', {})
        new_dic = {k[1:]: v for k, v in new_dic.items()}
        return new_dic

    @classmethod
    def _cur_kv_str(cls, k, v, cur_dic):
        # special key
        i1 = float('inf')
        if '.' in k:
            i1 = k.index('.')

        i2 = float('inf')
        if '[' in k and ']' in k:
            i2 = k.index('[')

        i, flag = min((i1, 1), (i2, 2), key=lambda x: x[0])
        if i == float('inf'):
            pass

        elif flag == 1:  # k = 'a.b'
            a, b = k.split('.', 1)
            v = cls._cur_kv_str(b, v, cur_dic.get(a, {}))
            return {a: v} if a != '' else v

        elif flag == 2:  # k = 'a[0]'
            a, i = k.split('[', 1)
            i, b = i.split(']', 1)
            i = int(i)
            cur_list = cur_dic.get(a, [])
            cur_list += [{}] * (i - len(cur_list) + 1)
            v = cls._cur_kv_str(b, v, cur_list[i])
            cur_list[i] = cls.merge_dict(cur_list[i], v)
            return {a: cur_list} if a != '' else cur_list

        # special value
        if isinstance(v, dict):  # v = {'a.b': 1}
            cur_dic[k] = cls._cur_dict(v, cur_dic.get(k, {}))
            return cur_dic

        else:
            if isinstance(v, str) and '=' in v:  # v = 'a.b=1'
                kk, vv = v.split('=', 1)
                kk, vv = kk.strip(), vv.strip()
                if cls.convert_to_constant_value:
                    vv = converter.DataConvert.str_to_constant(vv)
                v = cls._cur_dict({kk: vv}, cur_dic.get(k, {}))
            else:  # v = 'a=1'
                if cls.convert_to_constant_value:
                    v = converter.DataConvert.complex_str_to_constant(v)
            cur_dic[k] = v
            return cur_dic

    @classmethod
    def _cur_dict(cls, cur_dic, new_dic):
        for k, v in cur_dic.items():
            new_dic = cls.merge_dict(new_dic, cls._cur_kv_str(k, v, new_dic))

        return new_dic

    @classmethod
    def expand_dict(cls, d: dict):
        """expand dict while '.' in key or '=' in value

        Example:
            >>> d = {'a.b': 1}
            >>> ConfigObjParse.expand_dict(d)
            {'a': {'b': 1}}

            >>> d = {'a': 'b=1'}
            >>> ConfigObjParse.expand_dict(d)
            {'a': {'b': 1}}

            >>> d = {'a[1].b1.c': 1, 'a[0].b0.c': 0, 'a[1].b1.d': 2}
            >>> ConfigObjParse.expand_dict(d)
            {'a': [{'b0': {'c': 0}}, {'b1': {'c': 1, 'd': 2}}]}

            >>> d = {'a.b.c.d': 1, 'a.b': 'c.e=2', 'a.b.e': 3}
            >>> ConfigObjParse.expand_dict(d)
            {'a': {'b': {'c': {'d': 1, 'e': '2'}, 'e': 3}}}
        """
        return cls._cur_dict(d, {})

    @classmethod
    def expand_list_str(cls, d: List[str]):
        """expand dict while '.' in key or '=' in value

        Example:
            >>> d = ['a.b=1']
            >>> ConfigObjParse.expand_list_str(d)
            {'a': {'b': 1}}

            >>> d = ['a.b.c.d=1', 'a.b.c.e=2', 'a.b.e=3']
            >>> ConfigObjParse.expand_list_str(d)
            {'a': {'b': {'c': {'d': 1, 'e': '2'}, 'e': 3}}}
        """
        ret = {}
        for s in d:
            cls._cur_kv_str('', s, ret)
        return ret.get('', {})

    @staticmethod
    def merge_dict(d1: dict, d2: dict, depth=None) -> dict:
        """merge values from d1 and d2
        if had same key, d2 will cover d1

        Example:
            >>> d1 = {'a': {'b': {'c': 1}}}
            >>> d2 = {'a': {'b': {'d': 2}}}
            >>> ConfigObjParse.merge_dict(d1, d2)
            {'a': {'b': {'c': 1, 'd': 2}}}

            >>> d1 = {'a': [{'b1': 1}, {'c1': 1}]}
            >>> d2 = {'a': [{'b2': 2}, {'c2': 2}]}
            >>> ConfigObjParse.merge_dict(d1, d2)
            {'a': [{'b1': 1, 'b2': 2}, {'c1': 1, 'c2': 2}]}

        """

        def cur(cur_dic, new_dic, cur_depth=0):
            if depth and cur_depth > depth:
                return new_dic

            cur_dic = {**cur_dic}  # copy, don't change origin
            for k, v1 in new_dic.items():
                if k not in cur_dic:
                    pass
                elif isinstance(v1, dict) and isinstance(cur_dic[k], dict):
                    v2 = cur_dic[k]
                    v1 = cur(v2, v1, cur_depth=cur_depth + 1)
                elif isinstance(v1, list) and isinstance(cur_dic[k], list):
                    v2 = cur_dic[k]
                    for i, (vv1, vv2) in enumerate(zip(v1, v2)):
                        if isinstance(v1, dict) and isinstance(cur_dic[k], dict):
                            v1[i] = cur(vv2, vv1, cur_depth=cur_depth + 1)
                        else:
                            v1[i] = [vv2, vv1]

                cur_dic[k] = v1

            return cur_dic

        if not d2:
            return d1

        if not d1:
            return d2

        return cur(d1, d2)

    @classmethod
    def parse_config_obj_example(cls, config_path, parser) -> dict:
        """an example for parse parameters"""

        def params_params_from_file(path) -> dict:
            """user params, low priority"""
            return cls.expand_dict(os_lib.loader.load_yaml(path))

        def params_params_from_env(flag='Global.') -> dict:
            """global params, middle priority"""
            import os

            args = {}
            for k, v in os.environ.items():
                if k.startswith(flag):
                    k = k.replace(flag, '')
                    args[k] = v

            config = cls.expand_dict(args)
            config = converter.DataConvert.complex_str_to_constant(config)

            return config

        def params_params_from_arg(parser) -> dict:
            """local params, high priority
            # parser will be created like that
            import argparse

            parser = argparse.ArgumentParser()
            ...
            parser.add_argument('-c', '--config', nargs='+', default=[], help='global config')
            """

            args = parser.parse_args()
            _config = args.config
            if _config:
                _config = dict(s.split('=') for s in _config)
                _config = cls.expand_dict(_config)
                _config = converter.DataConvert.complex_str_to_constant(_config)
            else:
                _config = {}

            return _config

        config = params_params_from_file(config_path)
        config = cls.merge_dict(config, params_params_from_env())
        config = cls.merge_dict(config, params_params_from_arg(parser))

        return config


def permute_obj(obj: dict or list):
    """

    Example:

        >>> kwargs = [{'a': [1], 'b': [2, 3]}, {'c': [4, 5, 6]}]
        >>> permute_obj(kwargs)
        [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'c': 4}, {'c': 5}, {'c': 6}]

    """

    def cur(cur_obj: dict):
        r = [{}]
        for k, v in cur_obj.items():
            r = [{**rr, k: vv} for rr in r for vv in v]

        return r

    ret = []
    if isinstance(obj, dict):
        ret += cur(obj)
    else:
        for o in obj:
            ret += cur(o)

    return ret


def default(*args):
    """check the items by order and return the first item which is not None"""
    for obj in args:
        if obj is not None:
            return obj


def execute_cmd(cmd, **run_kwargs):
    default_kwargs = dict(
        args=cmd,
        shell=True,
        errors='ignore',
    )

    run_kwargs = ConfigObjParse.merge_dict(default_kwargs, run_kwargs)
    result = subprocess.run(**run_kwargs)
    return result


class PydanticParse:

    @classmethod
    def parse_model(cls, model: 'pydantic.BaseModel'):
        """
        Usage:
            .. code-block:: python

                class F1(pydantic.BaseModel):
                    aa: str


                class F2(pydantic.BaseModel):
                    a: str
                    b: str = 'b'
                    c: F1 | dict = {}
                    d: List[F1]

                PydanticParse.parse_model(F2)
                # {'a': {'type': 'str', 'is_required': True}, 'b': {'type': 'str', 'is_required': False, 'default': 'b'}, 'c': {'type': '__main__.F1 | dict', 'is_required': False, 'default': {}}, 'd': {'type': 'List[F1]', 'is_required': True}, 'd[].aa': {'type': 'str', 'is_required': True}}

        """
        result = {}

        fields = cls.get_model_fields(model)
        cls.process_fields(result, fields)

        return result

    @classmethod
    def process_fields(cls, result, fields, prefix=""):
        import pydantic

        for field_name, field_info in fields.items():
            full_name = f"{prefix}{field_name}" if prefix else field_name
            field_type = field_info.annotation if hasattr(field_info, 'annotation') else field_info.type_
            type_name = cls.get_type_name(field_type)
            is_required = field_info.is_required() if hasattr(field_info, 'is_required') else field_info.field_info.required
            desc = field_info.description if hasattr(field_info, 'description') else field_info.field_info.description

            field_data = {
                "type": type_name,
                "is_required": is_required,
            }

            if desc is not None:
                field_data['desc'] = desc

            if not is_required:
                default_val = field_info.default_factory() if hasattr(field_info, 'default_factory') and field_info.default_factory else field_info.default
                if default_val is not None:
                    field_data["default"] = default_val

            result[full_name] = field_data

            if issubclass(field_type, pydantic.BaseModel):
                nested_fields = cls.get_model_fields(field_type)
                cls.process_fields(result, nested_fields, f"{full_name}.")
            elif cls.is_list_of_pydantic(field_type):
                inner_type = get_args(field_type)[0]
                if issubclass(inner_type, pydantic.BaseModel):
                    nested_fields = cls.get_model_fields(inner_type)
                    cls.process_fields(result, nested_fields, f"{full_name}[].")

        return result

    @classmethod
    def get_model_fields(cls, model_class):
        if hasattr(model_class, 'model_fields'):
            # Pydantic v2
            return model_class.model_fields
        else:
            # Pydantic v1
            return model_class.__fields__

    @classmethod
    def get_type_name(cls, field_type):
        origin = get_origin(field_type)
        if origin is list:
            inner_type = get_args(field_type)[0]
            return f"List[{inner_type.__name__}]"
        elif origin is dict:
            key_type, value_type = get_args(field_type)
            return f"Dict[{key_type.__name__}, {value_type.__name__}]"
        else:
            if hasattr(field_type, '__name__'):
                return field_type.__name__
            else:
                return str(field_type)

    @classmethod
    def is_list_of_pydantic(cls, field_type):
        origin = get_origin(field_type)
        if origin is not list:
            return False
        args = get_args(field_type)
        if not args:
            return False
        return True
