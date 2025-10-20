"""utils for convert one object(like data structure of python, instance of python, file of os, etc.) to another object"""
import base64
import hashlib
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from .op_utils import IgnoreException
from .os_lib import FakeIo

ignore_exception = IgnoreException(stdout_method=FakeIo())


class DataConvert:
    """convert one data structure to another data structure"""

    @classmethod
    def custom_to_constant(cls, obj: Optional[list | dict | np.ndarray]):
        """convert a custom type(like np.ndarray) to a constant type(like int, float, str)
        recurse to convert the obj

        >>>DataConvert.custom_to_constant({0: [np.array([1, 2, 3]), np.array([4, 5, 6])]})
        {0: [[1, 2, 3], [4, 5, 6]]}

        """
        if isinstance(obj, dict):
            obj = cls.dict_to_constant(obj)
        elif isinstance(obj, list):
            obj = cls.list_to_constant(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
            obj = cls.img_array_to_constant(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype != np.uint8:
            obj = cls.np_to_constant(obj)
        elif isinstance(obj, (bool, int, float)):
            pass
        else:
            obj = str(obj)

        return obj

    @classmethod
    def dict_to_constant(cls, obj: dict):
        for k, v in obj.items():
            obj[k] = cls.custom_to_constant(v)

        return obj

    @classmethod
    def list_to_constant(cls, obj: list):
        for i, e in enumerate(obj):
            obj[i] = cls.custom_to_constant(e)
        return obj

    @staticmethod
    def np_to_constant(obj: np.ndarray):
        if len(obj.shape) == 1 and obj.size == 1:
            if isinstance(obj, (np.float32, np.float64)):
                obj = float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                obj = int(obj)
        else:
            obj = obj.tolist()

        return obj

    @classmethod
    def img_array_to_constant(cls, obj: np.ndarray) -> str:
        return cls.image_to_base64(obj)

    @classmethod
    def bytes_to_custom(cls, obj: bytes):
        """bytes to custom type(like np.ndarray, list)"""

    @staticmethod
    def image_to_bytes(obj: np.ndarray, fmt='.png') -> bytes:
        return cv2.imencode(fmt, obj)[1].tobytes()

    @staticmethod
    def bytes_to_image(obj: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(obj, dtype=np.uint8), -1)

    @classmethod
    def image_to_base64(cls, obj: np.ndarray, add_prefix=False, fmt='png') -> str:
        obj = cls.image_to_bytes(obj)
        obj = cls.bytes_to_base64(obj)
        if add_prefix:
            obj = f'data:image/{fmt};base64,{obj}'
        return obj

    @classmethod
    def base64_to_image(cls, obj: str) -> np.ndarray:
        obj = re.sub('data:image/.*;base64,', '', obj)
        obj = cls.base64_to_bytes(obj)
        return cls.bytes_to_image(obj)

    @staticmethod
    def bytes_to_base64(obj: bytes) -> str:
        return str(base64.b64encode(obj), 'utf-8')

    @staticmethod
    def base64_to_bytes(obj: str) -> bytes:
        return base64.b64decode(obj)

    @classmethod
    def file_to_base64(cls, obj: str or Path):
        with open(obj, 'rb') as f:
            return cls.bytes_to_base64(f.read())

    @classmethod
    def complex_str_to_constant(cls, obj: dict or list or str):
        """
        >>> DataConvert.complex_str_to_constant({0: '1', 1: '1.0', 2: 'true', 3: 'abc'})
        {0: 1, 1: 1.0, 2: 1, 3: 'abc'}

        """
        if isinstance(obj, dict):
            obj = {k: cls.complex_str_to_constant(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            obj = [cls.complex_str_to_constant(v) for v in obj]

        elif isinstance(obj, str):
            obj = cls.str_to_constant(obj)
            if isinstance(obj, (dict, list)):
                obj = cls.complex_str_to_constant(obj)

        return obj

    @classmethod
    def str_to_constant(cls, obj: str):
        """
        >>> DataConvert.str_to_constant('1')
        1
        >>> DataConvert.str_to_constant('1.0')
        1.0
        >>> DataConvert.str_to_constant('true')
        1
        >>> DataConvert.str_to_constant('abc')
        'abc'

        """
        obj = cls.str_to_constant_str(obj)
        for func in [cls.str_to_int, cls.str_to_float, cls.str_to_bool, cls.str_to_complex_constant]:
            s = func(obj)
            if s is not None:
                return s

        return obj

    @staticmethod
    def str_to_constant_str(obj: str):
        """filter '' or "" in str"""
        if obj[0] in ['"', "'"] and obj[-1] in ['"', "'"]:
            obj = obj[1:-1]
        return obj

    @staticmethod
    @ignore_exception.add_ignore(err_type=ValueError)
    def str_to_int(obj):
        return int(obj)

    @staticmethod
    @ignore_exception.add_ignore(err_type=ValueError)
    def str_to_float(obj):
        return float(obj)

    @staticmethod
    @ignore_exception.add_ignore(err_type=ValueError)
    def str_to_bool(obj):
        obj = obj.lower()
        if obj in ('y', 'yes', 't', 'true', 'on', '1'):
            return True
        elif obj in ('n', 'no', 'f', 'false', 'off', '0'):
            return False
        else:
            raise ValueError("invalid truth value %r" % (obj,))

    @staticmethod
    @ignore_exception.add_ignore(err_type=(json.decoder.JSONDecodeError, TypeError))
    def str_to_complex_constant(obj) -> list or dict:
        return json.loads(obj)

    @classmethod
    def obj_to_md5(cls, obj):
        if isinstance(obj, bytes):
            return cls.bytes_to_md5(obj)
        elif isinstance(obj, str):
            return cls.str_to_md5(obj)
        elif isinstance(obj, dict):
            return cls.dict_to_md5(obj)
        elif isinstance(obj, Path):
            return cls.file_to_md5(obj)
        elif isinstance(obj, Iterable):
            return cls.large_bytes_to_md5(obj)
        else:
            return cls.str_to_md5(str(obj))

    @staticmethod
    def bytes_to_md5(obj: bytes):
        return hashlib.md5(obj).hexdigest()

    @classmethod
    def str_to_md5(cls, obj: str):
        return cls.bytes_to_md5(obj.encode(encoding='utf8'))

    @classmethod
    def dict_to_md5(cls, obj: dict, sort=False):
        if sort:
            sort_keys = sorted(obj.keys())
            obj = {k: obj[k] for k in sort_keys}
        return cls.str_to_md5(str(obj))

    @classmethod
    def file_to_md5(cls, obj: str or Path):
        with open(obj, 'rb') as f:
            return cls.bytes_to_md5(f.read())

    @classmethod
    def large_file_to_md5(cls, obj: str or Path, chunk_size=1024 * 8):
        md5_hash = hashlib.md5()
        with open(obj, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    @classmethod
    def large_bytes_to_md5(cls, obj: Iterable[bytes]):
        md5_hash = hashlib.md5()
        for chunk in obj:
            md5_hash.update(chunk)
        return md5_hash.hexdigest()

    @staticmethod
    def file_to_sha256(obj: str or Path, chunk_size=1024 * 1024):
        hash_sha256 = hashlib.sha256()

        with open(obj, "rb") as f:
            # avoid to read big file
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    @staticmethod
    def time_to_timestamp(time_str: str, fmt="%H:%M:%S.%f") -> float:
        time_obj = datetime.strptime(time_str, fmt)

        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000.0
        return total_seconds

    @staticmethod
    def timestamp_to_time(timestamp: int | float, fmt="%H:%M:%S.%f") -> str:
        time_obj = datetime.utcfromtimestamp(timestamp)
        time_str = time_obj.strftime(fmt)
        return time_str


class DataInsConvert:
    """convert one data structure to a python instance"""

    @staticmethod
    def str_to_instance(obj: str):
        """give a str, return a class instance
        >>> DataInsConvert.str_to_instance('utils.DataInsConvert')
        """
        import importlib
        module, cls = obj.rsplit('.', 1)
        return getattr(importlib.import_module(module, package=None), cls)

    @staticmethod
    def instance_to_str(obj):
        pass

    @staticmethod
    def bytes_to_BytesIO(obj: bytes):
        """give a bytes, return a BytesIO instance contained the bytes"""
        f = io.BytesIO()
        f.write(obj)
        return f


class FileConvert:
    """convert one file to another file(s)"""

    @staticmethod
    def tar_to_zip(in_path: str, out_path: str):
        pass

    @staticmethod
    def pdf_to_images(in_path: str, out_dir: str):
        pass

    @staticmethod
    def images_to_pdf(in_dir: str, out_path: str):
        pass
