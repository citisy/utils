"""utils for configs, usually used in project startup, function definition, etc."""
import copy
from typing import List

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

        elif flag == 1:    # k = 'a.b'
            a, b = k.split('.', 1)
            v = cls._cur_kv_str(b, v, cur_dic.get(a, {}))
            return {a: v} if a != '' else v

        elif flag == 2:     # k = 'a[0]'
            a, i = k.split('[', 1)
            i, b = i.split(']', 1)
            i = int(i)
            cur_list = cur_dic.get(a, [])
            cur_list += [{}] * (i - len(cur_list) + 1)
            v = cls._cur_kv_str(b, v, cur_list[i])
            cur_list[i] = cls.merge_dict(cur_list[i], v)
            return {a: cur_list} if a != '' else cur_list

        # special value
        if isinstance(v, dict):     # v = {'a.b': 1}
            cur_dic[k] = cls._cur_dict(v, cur_dic.get(k, {}))
            return cur_dic

        else:
            if isinstance(v, str) and '=' in v:     # v = 'a.b=1'
                kk, vv = v.split('=', 1)
                kk, vv = kk.strip(), vv.strip()
                if cls.convert_to_constant_value:
                    vv = converter.DataConvert.str_to_constant(vv)
                v = cls._cur_dict({kk: vv}, cur_dic.get(k, {}))
            else:   # v = 'a=1'
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
    def merge_dict(d1: dict, d2: dict) -> dict:
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

        def cur(cur_dic, new_dic):
            for k, v1 in new_dic.items():
                if k not in cur_dic:
                    pass
                elif isinstance(v1, dict) and isinstance(cur_dic[k], dict):
                    v2 = cur_dic[k]
                    v1 = cur(v2, v1)
                elif isinstance(v1, list) and isinstance(cur_dic[k], list):
                    v2 = cur_dic[k]
                    for i, (vv1, vv2) in enumerate(zip(v1, v2)):
                        if isinstance(v1, dict) and isinstance(cur_dic[k], dict):
                            v1[i] = cur(vv2, vv1)
                        else:
                            v1[i] = [vv2, vv1]

                cur_dic[k] = v1

            return cur_dic

        return cur(copy.deepcopy(d1), copy.deepcopy(d2))

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


class PydanticParse:
    @classmethod
    def parse_model(cls, model: 'pydantic.BaseModel', return_example=False, return_default_value=False) -> dict:
        """
        Usage:
            .. code-block:: python

                class F1(pydantic.BaseModel):
                    aa: str

                class F2(pydantic.BaseModel):
                    a: str
                    b: str = 'b'
                    c: F1 = {}

                parse_pydantic(F2)
                # {'a': {'types': ['string'], 'is_required': True}, 'b': {'types': ['string'], 'is_required': False, 'default': 'b'}, 'c': {'types': [{'aa': {'types': ['string'], 'is_required': True}}], 'is_required': False, 'default': {}}}

                parse_pydantic(F2, return_example=True)
                # {'a': 'string', 'b': 'string', 'c': {'aa': 'string'}}

                parse_pydantic(F2, return_example=True, return_default_value=True)
                # {'a': 'string', 'b': 'b', 'c': {'aa': 'string'}}

        """

        import pydantic
        schema = model.schema()

        if pydantic.__version__ < '2':
            definitions = schema.get('definitions', {})
        else:
            definitions = schema.get('$defs', {})

        ret = cls.parse_schema(schema, definitions)
        if return_example:
            ret = cls.parse_ret_dict(ret, return_default_value)

        return ret

    @classmethod
    def parse_schema(cls, schema: dict, definitions={}) -> dict:
        ret = {}
        required = schema.get('required', [])
        for name, attr in schema['properties'].items():
            types = cls.parse_properties(attr)
            for i, _type in enumerate(types):
                if isinstance(_type, dict):
                    for v in _type.values():
                        for ii, __type in enumerate(v['types']):
                            if __type in definitions:
                                v['types'][ii] = cls.parse_schema(definitions[__type], definitions)
                else:
                    if _type in definitions:
                        types[i] = cls.parse_schema(definitions[_type], definitions)

            ret[name] = dict(
                types=types,
                is_required=name in required,
            )

            if 'default' in attr:
                # support for pydantic.__version__ > '2'
                ret[name]['default'] = attr['default']

            elif name not in required:
                ret[name]['default'] = None

        return ret

    @staticmethod
    def parse_properties(attr: dict) -> list:
        def parse(a):
            if 'type' in a:
                _type = a['type']
            elif '$ref' in a:
                obj = a['$ref']
                _type = obj.split('/')[-1]
            else:
                _type = ''
            return _type

        types = []
        tmp = types
        a = attr
        while 'items' in a or 'additionalProperties' in a or 'allOf' in a:
            if 'items' in a:
                _type = parse(a)
                tmp.append(_type)
                a = a['items']

            elif 'additionalProperties' in a:
                _type = parse(a)
                _tmp = []
                tmp.append({_type: dict(types=_tmp, is_required=True)})
                tmp = _tmp
                a = a['additionalProperties']

            elif 'allOf' in a:
                _type = parse(a['allOf'][0])
                a = a['allOf'][0]

        _type = parse(a)
        if _type:
            tmp.append(_type)

        return types

    @classmethod
    def parse_ret_dict(cls, ret: dict, return_default_value=False, return_na_value=False) -> dict:
        """
        >>> ret = {'a': {'types': ['string'], 'is_required': True}, 'b': {'types': ['string'], 'is_required': False, 'default': 'b'}, 'c': {'types': [{'aa': {'types': ['string'], 'is_required': True}}], 'is_required': False, 'default': {}}}
        >>> PydanticParse.parse_ret_dict(ret)
        {'a': 'string', 'b': 'string', 'c': {'aa': 'string'}}
        """
        d = {}
        for k, v in ret.items():
            a = []
            b = a
            for _type in v['types']:
                if isinstance(_type, dict):
                    b.append(cls.parse_ret_dict(_type, return_default_value=return_default_value))
                elif _type == 'array':
                    if 'default' in v:
                        b.append(v['default'])
                        break
                    else:
                        c = []
                        b.append(c)
                        b = c
                elif return_default_value and 'default' in v:
                    b.append(v['default'])
                else:
                    b.append(_type)

            if a:
                d[k] = a[0]
            elif return_default_value and 'default' in v:
                d[k] = v['default']
            elif return_na_value:
                d[k] = None

        return d
