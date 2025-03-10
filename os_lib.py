"""utils for loading data from operating system(like file system, database, etc.) to memory,
or saving data from memory to operating system"""
import configparser
import json
import os
import pickle
import random
import re
import time
import uuid
from pathlib import Path
from typing import List, Iterable

import cv2
import numpy as np
import pandas as pd
import yaml     # pip install PyYAML


def mk_dir(dir_path):
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=True)


def mk_parent_dir(file_path):
    file_path = Path(file_path)
    dir_path = file_path.parent
    mk_dir(dir_path)


suffixes_dict = dict(
    json=('.json', '.js'),
    jsonl=('.jsonl', '.jsl'),
    yml=('.yml', '.yaml'),
    ini=('.ini',),
    txt=('.txt',),
    pkl=('.pkl',),
    joblib=('.joblib',),
    img=('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
    csv=('.csv', '.tsv'),
    excel=('.xlsx', '.xls'),
    word=('.docx', '.doc'),
    pdf=('.pdf',),
    npy=('.npy',),
    npz=('.npz',)
)


def auto_suffix(obj):
    if isinstance(obj, (list, tuple, set, str)):
        s = suffixes_dict['txt'][0]
    elif isinstance(obj, dict):
        s = suffixes_dict['json'][0]
    elif isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
        s = suffixes_dict['img'][0]
    elif isinstance(obj, np.ndarray):
        s = suffixes_dict['npy'][0]
    elif isinstance(obj, pd.DataFrame):
        s = suffixes_dict['csv'][0]
    else:
        s = suffixes_dict['pkl'][0]
    return s


def find_all_suffixes_files(root_dir, suffixes):
    return (p for p in Path(root_dir).glob('*') if p.suffix.lower() in suffixes)


class Saver:
    def __init__(self, verbose=False, stdout_method=print, stdout_fmt='Succeed to save to %s !', stderr_method=print, stderr_fmt='Fail to save to %s !'):
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.stdout_fmt = stdout_fmt
        self.stderr_method = stderr_method
        self.stderr_fmt = stderr_fmt

        self.funcs = {
            suffixes_dict['json']: self.save_json,
            suffixes_dict['jsonl']: self.save_jsonl,
            suffixes_dict['yml']: self.save_yml,
            suffixes_dict['ini']: self.save_ini,
            suffixes_dict['txt']: self.save_txt,
            suffixes_dict['pkl']: self.save_pkl,
            suffixes_dict['joblib']: self.save_joblib,
            suffixes_dict['img']: self.save_img,
            suffixes_dict['npy']: self.save_npy,
            suffixes_dict['npz']: self.save_npz,
            suffixes_dict['csv']: self.save_csv,
            suffixes_dict['excel']: self.save_excel
        }

    def stdout(self, path):
        self.stdout_method(self.stdout_fmt % path)

    def stderr(self, path):
        self.stderr_method(self.stderr_fmt % path)

    def auto_save(self, obj, path: str, **kwargs):
        suffix = Path(path).suffix.lower()
        mk_parent_dir(path)

        for k, func in self.funcs.items():
            if suffix in k:
                func(obj, path, **kwargs)
                break
        else:
            self.save_bytes(obj, path, **kwargs)

    def save_json(self, obj: dict, path, **kwargs):
        kwargs.setdefault('ensure_ascii', False)
        kwargs.setdefault('indent', 4)
        with open(path, 'w', encoding='utf8', errors='ignore') as f:
            json.dump(obj, f, **kwargs)

        self.stdout(path)

    def save_jsonl(self, obj: List[dict], path, **kwargs):
        obj = [json.dumps(o, ensure_ascii=False) for o in obj]
        self.save_txt(obj, path, **kwargs)
        self.stdout(path)

    def save_yml(self, obj: dict, path, **kwargs):
        with open(path, 'w') as f:
            yaml.dump(obj, f)
        self.stdout(path)

    def save_ini(self, obj: dict, path, **kwargs):
        parser = configparser.ConfigParser()
        parser.read_dict(obj)

        # 将ConfigParser对象写入文件
        with open(path, 'w', encoding='utf8') as f:
            parser.write(f)

        self.stdout(path)

    def save_txt(self, obj: Iterable[str] | str, path, sep='\n', **kwargs):
        """if obj is str, write the str to file directly,
        if obj is list, write line by line with sep token"""
        with open(path, 'w', encoding='utf8', errors='ignore') as f:
            if isinstance(obj, str):
                f.write(obj)
            else:
                for o in obj:
                    f.write(f'{o}{sep}')

        self.stdout(path)

    def save_pkl(self, obj, path, **kwargs):
        with open(path, 'wb') as f:
            pickle.dump(obj, f, **kwargs)

        self.stdout(path)

    def save_joblib(self, obj, path, **kwargs):
        import joblib
        joblib.dump(obj, path, **kwargs)

        self.stdout(path)

    def save_bytes(self, obj: bytes, path, **kwargs):
        with open(path, 'wb') as f:
            f.write(obj)

        self.stdout(path)

    def save_excel(self, obj: pd.DataFrame, path, **kwargs):
        obj.to_excel(path, **kwargs)
        self.stdout(path)

    def save_img(self, obj: np.ndarray, path, **kwargs):
        # it will error with chinese path in low version of cv2
        # it has fixed in high version already
        cv2.imencode(Path(path).suffix, obj)[1].tofile(path)
        self.stdout(path)
        # flag = cv2.imwrite(path, obj)
        # if flag:
        #     self.stdout(path)
        # else:
        #     self.stderr(path)

    def save_npy(self, obj: np.ndarray, path, **kwargs):
        np.save(path, obj, **kwargs)
        self.stdout(path)

    def save_npz(self, obj: List[np.ndarray], path, **kwargs):
        np.savez(path, *obj, **kwargs)
        self.stdout(path)

    def save_csv(self, obj: pd.DataFrame, path, **kwargs):
        obj.to_csv(path, **kwargs)
        self.stdout(path)

    def save_image_from_pdf(self, path, page=None, image_dir=None, scale_ratio=1.33):
        """select pages from pdf, and save with image type"""
        images = loader.load_images_from_pdf2(
            path,
            scale_ratio=scale_ratio
        )

        if isinstance(page, int):
            images = [images[page]]

        elif isinstance(page, list):
            images = [images[_] for _ in page]

        image_dir = image_dir or path.replace('pdfs', 'images').replace(Path(path).suffix, '')
        mk_dir(image_dir)

        for i, img in enumerate(images):
            self.save_img(img, f'{image_dir}/{i}.png')

    def save_pdf_from_pdf(self, path, page=None, save_path=None):
        """select pages from pdf, and save with pdf type"""
        from PyPDF2 import PdfFileReader, PdfFileWriter, errors

        try:
            pdf_reader = PdfFileReader(path)
            pdf_writer = PdfFileWriter()

            if isinstance(page, int):
                pdf_writer.addPage(pdf_reader.getPage(page))

            elif isinstance(page, list):
                for i in page:
                    pdf_writer.addPage(pdf_reader.getPage(i))

            suffix = Path(path).suffix
            save_path = save_path or str(path).replace(suffix, '_' + suffix)

            with open(save_path, 'wb') as out:
                pdf_writer.write(out)

            self.stdout(save_path)
        except errors.PdfReadError:
            self.stderr(save_path)
        except KeyError:
            self.stderr(save_path)

    def save_image_to_pdf(self, obj: bytes | str, path):
        import fitz  # pip install PyMuPDF

        if isinstance(obj, str):
            img_doc = fitz.open(obj)
        else:
            img_doc = fitz.open(stream=obj, filetype='png')

        img_pdf = fitz.open(stream=img_doc.convert_to_pdf(), filetype='pdf')

        with fitz.open() as doc:
            doc.insert_pdf(img_pdf)
            doc.save(path)
        self.stdout(path)

    def save_images_to_pdf(self, obj: List, path):
        import fitz  # pip install PyMuPDF

        doc = fitz.open()
        for img in obj:
            if isinstance(img, str):
                img_doc = fitz.open(img)
            else:
                img_doc = fitz.open(stream=img, filetype='png')

            img_pdf = fitz.open(stream=img_doc.convert_to_pdf(), filetype='pdf')
            doc.insert_pdf(img_pdf)
        doc.save(path)
        self.stdout(path)


class Loader:
    def __init__(self, verbose=False, stdout_method=print, stdout_fmt='Read %s successful!'):
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.stdout_fmt = stdout_fmt

        self.funcs = {
            suffixes_dict['json']: self.load_json,
            suffixes_dict['jsonl']: self.load_jsonl,
            suffixes_dict['yml']: self.load_yaml,
            suffixes_dict['ini']: self.load_ini,
            suffixes_dict['txt']: self.load_txt,
            suffixes_dict['pkl']: self.load_pkl,
            suffixes_dict['img']: self.load_img,
            suffixes_dict['npy']: self.load_np_array,
            suffixes_dict['npz']: self.load_np_array,
            suffixes_dict['excel']: self.load_excel
        }

    def stdout(self, path):
        self.stdout_method(self.stdout_fmt % path)

    def auto_load(self, path: str):
        suffix = Path(path).suffix.lower()

        for k, func in self.funcs.items():
            if suffix in k:
                obj = func(path)
                break
        else:
            obj = self.load_bytes(path)

        return obj

    def load_json(self, path) -> dict:
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            obj = json.load(f)
        self.stdout(path)

        return obj

    def load_jsonl(self, path) -> List[dict]:
        obj = self.load_txt(path)
        obj = [json.loads(o) for o in obj]

        self.stdout(path)

        return obj

    def load_yaml(self, path) -> dict:
        obj = yaml.load(open(path, 'rb'), Loader=yaml.Loader)
        self.stdout(path)

        return obj

    def load_ini(self, path) -> configparser.ConfigParser:
        obj = configparser.ConfigParser()
        obj.read(path, encoding="utf-8")
        self.stdout(path)

        return obj

    def load_txt(self, path, split_line=True, sep='\n') -> str | Iterable[str]:
        # todo: maybe create a new function called load_txtl, like josnl?
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            obj = f.read().rstrip()
            if split_line:
                obj = obj.split(sep)

        self.stdout(path)
        return obj

    def load_txt_dir(self, dirt: str | Path, fmt='*.txt'):
        txts = []
        for fp in Path(dirt).glob(fmt):
            txts.append(self.load_txt(fp))
        return txts

    def load_pkl(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        self.stdout(path)
        return obj

    def load_bytes(self, path) -> bytes:
        with open(path, 'rb') as f:
            obj = f.read()

        self.stdout(path)
        return obj

    def load_excel(self, path, **kwargs) -> pd.DataFrame:
        df = pd.read_excel(path, **kwargs)
        self.stdout(path)
        return df

    def load_img(self, path, channel_fixed_3=False) -> np.ndarray:
        # it will error with chinese path in low version of cv2
        # it has fixed in high version already
        # but still bugs with `cv2.imread`, so still use the following method to read images
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        # img = cv2.imread(path)
        assert img is not None

        if channel_fixed_3:
            if img.shape[2] == 3:
                return img
            elif img.shape[2] == 4:  # bgra
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
            elif img.shape[2] == 1:  # gray
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return img
            else:
                raise ValueError("image has weird channel number: %d" % img.shape[2])

        self.stdout(path)
        return img

    def load_image_from_zipfile(self, path, zip_file):
        from .converter import DataConvert

        image = zip_file.open(path).read()
        image = DataConvert.bytes_to_image(image)

        self.stdout(path)
        return image

    def load_images_from_pdf(
            self,
            obj: str | bytes, scale_ratio: float = 1.33,
            rotate: int = 0, alpha=False, bgr=False
    ) -> List[np.ndarray]:
        import fitz  # pip install PyMuPDF

        if isinstance(obj, str):
            doc = fitz.open(obj)
        else:
            doc = fitz.open(stream=obj, filetype='pdf')

        images = []

        for page in doc:
            trans = fitz.Matrix(scale_ratio, scale_ratio).prerotate(rotate)  # rotate means clockwise
            pm = page.get_pixmap(matrix=trans, alpha=alpha)
            data = pm.tobytes()  # bytes
            img_np = np.frombuffer(data, dtype=np.uint8)

            flag = 1 if bgr else -1
            img = cv2.imdecode(img_np, flags=flag)

            images.append(img)

        if isinstance(obj, str):
            self.stdout(obj)

        return images

    def load_images_from_pdf2(
            self,
            obj: str | bytes, scale_ratio: float = 1.33,
    ) -> List[np.ndarray]:
        # pip install pdf2image
        from pdf2image import convert_from_path, convert_from_bytes, exceptions

        dpi = 72 * scale_ratio
        try:
            if isinstance(obj, (str, Path)):
                images = convert_from_path(obj, dpi=dpi)
            else:
                images = convert_from_bytes(obj, dpi=dpi)
        except exceptions.PDFPageCountError:
            images = []

        images = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]

        if isinstance(obj, str):
            self.stdout(obj)

        return images

    def load_np_array(self, path):
        obj = np.load(path)
        self.stdout(path)
        return obj

    def load_zip(self, path):
        from zipfile import ZipFile

        objs = []
        with ZipFile(path, 'r') as zip_file:
            for fp in zip_file.namelist():
                if fp.endswith('/'):  # is dir
                    continue
                objs.append(zip_file.open(fp).read())

        self.stdout(path)
        return objs


class BaseCacher:
    RANDOM = 0
    OLDEST = 1
    LATEST = 2

    cache_stdout_fmt = 'Save _id[%s] successful!'
    delete_stdout_fmt = 'Delete _id[%s] successful!'
    get_stdout_fmt = 'Get _id[%s] successful!'

    def __call__(self, *args, **kwargs):
        return self.cache_one(*args, **kwargs)

    def cache_one(self, obj, **kwargs):
        raise NotImplemented

    def cache_batch(self, objs, **kwargs):
        raise NotImplemented

    def delete_one(self, **kwargs):
        raise NotImplemented

    def delete_batch(self, **kwargs):
        raise NotImplemented

    def delete_over_range(self, **kwargs):
        raise NotImplemented

    def get_one(self, **kwargs):
        raise NotImplemented

    def get_batch(self, **kwargs):
        raise NotImplemented


class Cacher(BaseCacher):
    def __init__(self, saver=None, loader=None, deleter=None, max_size=None):
        self.saver = saver
        self.loader = loader
        self.deleter = deleter
        self.max_size = max_size

    def cache_one(self, obj, _id=None, **kwargs):
        self.delete_over_range()
        return self.saver(obj, _id=_id)

    def cache_batch(self, objs, _ids=None, **kwargs):
        _ids = _ids or [None] * len(objs)
        return [self.cache_one(obj, _id=_id) for obj, _id in zip(objs, _ids)]

    def delete_one(self, _id=None, **kwargs):
        return self.deleter(_id)

    def delete_batch(self, _ids=(), **kwargs):
        return [self.delete_one(_id) for _id in _ids]

    def delete_over_range(self, **kwargs):
        if not self.max_size:
            return

        if len(self.saver) >= self.max_size:
            self.deleter()

    def get_one(self, _id=None, **kwargs):
        return self.loader(_id)

    def get_batch(self, _ids=(), **kwargs):
        return [self.get_one(_id) for _id in _ids]


class MemoryCacher(BaseCacher):
    def __init__(self, max_size=None,
                 verbose=True, stdout_method=print,
                 **saver_kwargs
                 ):
        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.cache = {}

    def cache_one(self, obj, _id=None, **kwargs):
        time_str = time.time()
        uid = str(uuid.uuid4())
        if _id is None:
            _id = (time_str, uid)
        self.delete_over_range()
        self.cache[_id] = obj
        self.stdout_method(self.cache_stdout_fmt % str(_id))
        return _id

    def cache_batch(self, objs, _ids=None, **kwargs):
        _ids = _ids or [None] * len(objs)
        ids = []
        for i, obj in enumerate(objs):
            _id = self.cache_one(obj)
            ids.append(_id)
        return ids

    def _search(self, keys, size=None, search_type=None):
        search_type = search_type or self.RANDOM
        size = 1 if size is None else size
        size = min(size, len(keys))

        if search_type == self.RANDOM:
            ids = np.random.choice(len(keys), size, replace=False)
            return [keys[i] for i in ids]

        elif search_type == self.OLDEST:
            keys = sorted(keys)
            return keys[:size]

        elif search_type == self.LATEST:
            keys = sorted(keys)
            return keys[-size:]

    def delete_one(self, _id=None, **kwargs):
        self.cache.pop(_id)
        self.stdout_method(self.delete_stdout_fmt % str(_id))

    def delete_batch(self, _ids=None, **kwargs):
        for _id in _ids:
            self.delete_one(_id=_id, **kwargs)

    def delete_over_range(self, **kwargs):
        if not self.max_size:
            return

        if len(self.cache) >= self.max_size:
            _ids = self._search(list(self.cache.keys()), len(self.cache) - self.max_size, search_type=self.OLDEST)
            self.delete_batch(_ids, **kwargs)

    def get_one(self, _id=None, key=None, search_type=None, **kwargs):
        if _id is None:
            keys = list(self.cache.keys())

            if key is not None:
                keys = [k for k in keys if key in k]

            _id = self._search(keys, search_type=search_type)
            if len(_id) == 0:
                return None
            _id = _id[0]

        self.stdout_method(self.get_stdout_fmt % str(_id))
        return self.cache.get(_id)

    def get_batch(self, _ids=None, key=None, size=None, search_type=None, **kwargs):
        if _ids is None:
            keys = list(self.cache.keys())

            if key is not None:
                keys = [k for k in keys if key in k]

            _ids = self._search(keys, size, search_type=search_type)

        self.stdout_method(self.get_stdout_fmt % str(_ids))
        return [self.cache.get(_id) for _id in _ids]


class FileCacher(BaseCacher):
    def __init__(self, cache_dir=None, max_size=None,
                 verbose=True, stdout_method=print,
                 ):
        mk_dir(cache_dir)
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.saver = Saver(verbose, stdout_method, stdout_fmt=self.cache_stdout_fmt)
        self.loader = Loader(verbose, stdout_method, stdout_fmt=self.get_stdout_fmt)

    def get_fn(self, obj=None, _id=None, file_name=None, file_stem=None):
        if _id is not None:
            file_name = _id
        elif file_name is not None:
            file_name = file_name
        elif file_stem is not None:
            file_name = str(file_stem) + auto_suffix(obj)
        else:
            file_name = str(uuid.uuid4()) + auto_suffix(obj)

        return file_name

    def cache_one(self, obj, _id=None, file_name=None, file_stem=None, **kwargs):
        file_name = self.get_fn(obj, _id, file_name, file_stem)
        path = f'{self.cache_dir}/{file_name}'
        self.delete_over_range(suffix=Path(path).suffix)
        self.saver.auto_save(obj, path)
        return file_name

    def cache_batch(self, objs, _ids=None, file_names=None, file_stems=None, **kwargs):
        _ids = _ids or [None] * len(objs)
        file_names = file_names or [None] * len(objs)
        file_stems = file_stems or [None] * len(objs)
        _fns = []
        for obj, _id, file_name, file_stem in zip(objs, _ids, file_names, file_stems):
            file_name = self.get_fn(obj, _id, file_name, file_stem)
            path = f'{self.cache_dir}/{file_name}'
            self.delete_over_range(suffix=Path(path).suffix)
            self.saver.auto_save(obj, path)
            _fns.append(file_name)
        return _fns

    def delete_over_range(self, suffix='', ignore_keys=(), **kwargs):
        if not self.max_size:
            return

        caches = [str(_) for _ in self.cache_dir.glob('*')]

        # python version > 3.9, use `Path.rglob` directly.
        p = re.compile(suffix)
        caches = [_ for _ in caches if p.search(_)]

        for key in ignore_keys:
            caches = [_ for _ in caches if key not in _]

        if len(caches) > self.max_size:
            try:
                ctime = [os.path.getctime(fp) for fp in caches]
                min_ctime = min(ctime)
                old_path = caches[ctime.index(min_ctime)]
                os.remove(old_path)
                self.stdout_method(self.delete_stdout_fmt % old_path)
                return old_path

            except FileNotFoundError:
                # todo: if it occur, number of file would be greater than max_size
                self.stdout_method('Two process thread were crashed while deleting file possibly')
                return

    def get_one(self, _id=None, file_name=None, **kwargs):
        file_name = _id or file_name

        if file_name is None:
            caches = [str(_) for _ in self.cache_dir.glob(f'*')]
            path = random.choice(caches)
        else:
            path = f'{self.cache_dir}/{file_name}'

        return self.loader.auto_load(path)

    def get_batch(self, _ids=None, file_names=None, size=None, **kwargs):
        if _ids is None and file_names is None:
            caches = [str(_) for _ in self.cache_dir.glob(f'*')]
            paths = np.random.choice(caches, size, replace=False)
        else:
            file_names = _ids or file_names
            paths = [f'{self.cache_dir}/{file_name}' for file_name in file_names]

        return [self.loader.auto_load(path) for path in paths]


class MongoDBCacher(BaseCacher):
    def __init__(self, host='127.0.0.1', port=27017, user=None, password=None, database=None, collection=None,
                 max_size=None, verbose=True, stdout_method=print,
                 **mongo_kwargs):
        from pymongo import MongoClient

        self.client = MongoClient(host, port, **mongo_kwargs)
        self.db = self.client[database]
        self.db.authenticate(user, password)
        self.collection = self.db[collection]
        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()

    def cache_one(self, obj: dict, _id=None, **kwargs):
        self.delete_over_range()

        obj.setdefault('update_time', int(time.time()))
        if _id is None:
            x = self.collection.insert_one(obj)
            _id = x.inserted_id

        else:
            self.collection.update_one({'_id': _id}, {'$set': obj}, upsert=True)

        self.stdout_method(self.cache_stdout_fmt % _id)
        return _id

    def cache_batch(self, objs, _ids=None, **kwargs):
        self.delete_over_range(len(objs))
        if _ids is None:
            x = self.collection.insert_many(objs)
            _ids = x.inserted_ids

        else:
            for _id, obj in zip(_ids, objs):
                self.collection.update_one({'_id': _id}, {'$set': obj}, upsert=True)

        return _ids

    def delete_over_range(self, batch_size=1, **kwargs):
        if not self.max_size:
            return

        query = self.collection.find()
        if query.count() > self.max_size - batch_size:
            x = query.sort({'update_time': 1}).limit(1)
            self.collection.delete_one(x)

    def get_one(self, _id=None, **kwargs):
        if _id is None:
            return self.collection.find_one()
        else:
            return self.collection.find_one({'_id': _id})

    def get_batch(self, _ids=None, size=None, **kwargs):
        if _ids is None:
            return [self.collection.find_one() for _ in range(size)]
        else:
            return [self.collection.find_one({'_id': _id}) for _id in _ids]


class RedisCacher(BaseCacher):
    def __init__(self, host='127.0.0.1', port=6379, db=0,
                 max_size=None, verbose=True, stdout_method=print,
                 **redis_kwargs) -> None:
        import redis

        self.client = redis.Redis(host=host, port=port, db=db, **redis_kwargs)
        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()

    def cache_one(self, obj: dict, _id=None, **kwargs):
        self.delete_over_range()

        s = int(time.time())
        obj.setdefault('update_time', s)
        if _id is None:
            _id = s

        self.client.hmset(_id, obj)
        self.stdout_method(self.cache_stdout_fmt % _id)
        return _id

    def cache_batch(self, objs, _ids=None, **kwargs):
        self.delete_over_range(len(objs))
        s = int(time.time())
        _ids = _ids or [f'{s}_{i}' for i in range(len(objs))]

        for _id, obj in zip(_ids, objs):
            obj.setdefault('update_time', s)
            self.client.hmset(_id, obj)
            self.stdout_method(self.cache_stdout_fmt % _id)

        return _ids

    def delete_over_range(self, batch_size=1, **kwargs):
        if not self.max_size:
            return

        if self.client.dbsize() > self.max_size - batch_size:
            raise NotImplementedError

    def get_one(self, _id=None, **kwargs):
        if _id is None:
            _id = self.client.randomkey()
        return self.client.hgetall(_id)

    def get_batch(self, _ids=None, size=None, **kwargs):
        rets = []
        if _ids is None:
            _ids = [self.client.randomkey() for _ in range(size)]
        for _id in _ids:
            rets.append(self.get_one(_id))
        return rets


class ESCacher(BaseCacher):
    """todo"""


class MySqlCacher(BaseCacher):
    host = '127.0.0.1'
    port = 3306
    user = None
    password = None
    database = None
    conn_kwargs = {}

    def __init__(
            self, table=None,
            max_size=None, verbose=True, stdout_method=print,
            **kwargs
    ):
        self.__dict__.update(kwargs)

        self.table = table

        _escape_table = [chr(x) for x in range(128)]
        _escape_table[0] = "\\0"
        _escape_table[ord("\\")] = "\\\\"
        _escape_table[ord("\n")] = "\\n"
        _escape_table[ord("\r")] = "\\r"
        _escape_table[ord("\032")] = "\\Z"
        _escape_table[ord('"')] = '\\"'
        _escape_table[ord("'")] = "\\'"
        self.escape_table = _escape_table

        self.max_size = max_size
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()

    @property
    def connection(self):
        # note, each time execute the sql, initialize a new connection, 'cause one connection would use the cache result
        import pymysql
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            **self.conn_kwargs
        )

    def cache_one(self, obj: dict, allow_duplicates=True, pri_key='id', **kwargs):
        if allow_duplicates or not kwargs:
            last_id = self._add(obj, **kwargs)

        else:
            data = self.get_one(**kwargs)
            if data:
                _ = self._update(obj, **kwargs)
                last_id = data[pri_key]
            else:
                last_id = self._add(obj, **kwargs)

        return last_id

    def std_value(self, v):
        if isinstance(v, (dict, list, tuple)):
            v = json.dumps(v, ensure_ascii=False)

        v = self.std_str_value(v)

        return v

    def std_str_value(self, v):
        if isinstance(v, str):
            v = v.translate(self.escape_table)
            v = f'"{v}"'

        return v

    def make_where_condition(self, obj: dict) -> list:
        where_conditions = []
        for k, v in obj.items():
            if isinstance(v, (list, tuple, set)):
                values = ''
                for vv in v:
                    vv = self.std_str_value(vv)
                    values += f'{vv},'
                values = values[:-1]
                where_conditions.append(f'{k} in ({values})')
            else:
                v = self.std_str_value(v)
                where_conditions.append(f'{k}={v}')

        return where_conditions

    def make_set_statements(self, obj: dict) -> list:
        set_statements = []
        for k, v in obj.items():
            v = self.std_value(v)
            set_statements.append(f'{k}={v}')

        return set_statements

    def _add(self, obj: dict, **kwargs):
        list_columns = []
        list_values = []

        if obj:
            for k, v in obj.items():
                list_columns.append(k)
                list_values.append(v)

        for k, v in kwargs.items():
            list_columns.append(k)
            list_values.append(v)

        columns = ','.join(list_columns)
        values = ''
        for v in list_values:
            v = self.std_value(v)
            values += f'{v},'
        values = values[:-1]

        sql = f'INSERT INTO {self.table} ({columns}) VALUES ({values})'

        with self.connection as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                last_id = cursor.lastrowid
                connection.commit()

        return last_id

    def _update(self, obj: dict, **kwargs):
        set_statements = self.make_set_statements(obj)
        set_statements = ', '.join(set_statements)

        where_conditions = self.make_where_condition(kwargs)
        where_conditions = ' and '.join(where_conditions)

        sql = f"UPDATE {self.table} SET {set_statements} WHERE {where_conditions}"

        with self.connection as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                last_id = cursor.lastrowid
                connection.commit()

        return last_id

    def get_one(self, additional_sql='', **kwargs) -> dict:
        """
        Usage:
            >>> MySqlCacher().get_one(id=0)
            >>> MySqlCacher().get_one(k1='s1', k2='s2')

        """
        where_conditions = self.make_where_condition(kwargs)
        where_conditions = ' and '.join(where_conditions)
        sql = f"select * from {self.table} where {where_conditions} {additional_sql} limit 1"

        with self.connection as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
                if row:
                    fields = [i[0] for i in cursor.description]
                    data = dict(zip(fields, row))
                else:
                    data = {}

        return data

    def get_batch(self, size=None, additional_sql='', **kwargs) -> List[dict]:
        """
        Usage:
            >>> MySqlCacher().get_batch(id=[0, 1])
            >>> MySqlCacher().get_batch(k1=['s1', 's2'], k2=['s3'])

        """
        where_conditions = self.make_where_condition(kwargs)
        where_conditions = ' and '.join(where_conditions)
        sql = f"select * from {self.table} where {where_conditions} {additional_sql}"
        if size:
            sql += f' limit {size}'

        with self.connection as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
                fields = [i[0] for i in cursor.description]
                data = [dict(zip(fields, row)) for row in result]

        return data


class PGSQLCacher(MySqlCacher):
    port = 5432

    def make_connection(self):
        import psycopg2  # pip install psycopg2-binary

        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            **self.conn_kwargs
        )

        return conn


class MilvusCacher(BaseCacher):
    def __init__(self, collection_name, **kwargs):
        from pymilvus import MilvusClient  # pip install pymilvus

        self.client = MilvusClient(**kwargs)
        self.collection_name = collection_name

    def make_example_data(self):
        des = self.client.describe_collection(self.collection_name)
        fields = des['fields']

    def cache_one(self, obj: dict, **kwargs):
        kwargs.setdefault('collection_name', self.collection_name)
        kwargs.setdefault('data', obj)
        res = self.client.insert(data=obj, **kwargs)
        return res['ids']

    def cache_batch(self, objs: List[dict], **kwargs):
        kwargs.setdefault('collection_name', self.collection_name)
        kwargs.setdefault('data', objs)
        res = self.client.insert(**kwargs)
        return res['ids']

    def delete_batch(self, **kwargs):
        kwargs.setdefault('collection_name', self.collection_name)
        return self.client.delete(**kwargs)

    def get_one(self, vector=None, is_search=True, **kwargs) -> dict:
        """
        Args:
            vector:
            is_search:
                True, use vector to query
                False, use key to query
            **kwargs:

        Usage:
            >>> MilvusCacher().get_batch(np.zeros(128))
            >>> MilvusCacher().get_one(filter='id==1', is_search=False)
            >>> MilvusCacher().get_one(ids=[1], is_search=False)

        """
        kwargs.setdefault('collection_name', self.collection_name)
        kwargs.setdefault('limit', 1)

        if is_search:
            if vector:
                kwargs.setdefault('data', [vector])

            res = self.client.search(**kwargs)[0]

        else:
            res = self.client.query(**kwargs)

        return res[0]

    def get_batch(self, vectors=None, is_search=True, size=None, **kwargs) -> List[List[dict]] | List[dict]:
        kwargs.setdefault('collection_name', self.collection_name)
        kwargs.setdefault('limit', size)
        if is_search:
            if vectors:
                kwargs.setdefault('data', vectors)
            res = self.client.search(**kwargs)

        else:
            res = self.client.query(**kwargs)

        return list(res)


class FakeIo:
    """a placeholder, empty io method to cheat some functions which must use an io method,
    it means that the method do nothing in fact,
    it is useful to reduce the number of code changes

    Examples
    .. code-block:: python

        # save obj
        io_method = open

        # do not save obj
        io_method = FakeIo

        with io_method(fp, 'w', encoding='utf8') as f:
            f.write(obj)
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)

    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, *args, **kwargs):
        pass


saver = Saver()
loader = Loader()
