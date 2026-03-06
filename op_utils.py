"""utils for operating instances of python, decorator usually"""
import time
from functools import wraps
from contextlib import contextmanager

from .os_lib import FakeIo


class IgnoreException:
    """
    Usage:
        .. code-block:: python

            ignore_exception = IgnoreException()

            @ignore_exception.add_ignore()
            def func():
                raise Exception

            @ignore_exception.add_ignore(error_message='there is an error')
            def func():
                raise Exception

            @ignore_exception.add_ignore(err_type=Exception)
            def func():
                raise Exception
    """

    def __init__(self, verbose=True, stdout_method=print, msg_fmt=None):
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.msg_fmt = msg_fmt or 'Ignore the error occur: {e}'

    @contextmanager
    def context(self, *args, e=None, **kwargs):
        msg = self.msg_fmt.format(e=e, **kwargs)
        self.stdout_method(msg)
        try:
            yield
        finally:
            pass

    def add_ignore(
            self,
            err_context=None,
            err_type=(ConnectionError, TimeoutError),
            raise_type=type(None),
            err_fn=None,
    ):
        context = err_context or self.context

        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                try:
                    return func(*args, **kwargs)

                except err_type as e:
                    if isinstance(e, raise_type):
                        raise e

                    with context(*args, e=e, **kwargs):
                        if err_fn:
                            return err_fn(*args, **kwargs)

            return wrap

        return wrap2


ignore_exception = IgnoreException()


class Retry:
    """
    Usage:
        .. code-block:: python

            retry = Retry()

            @retry.add_try()
            def func():
                raise Exception

            @retry.add_try(err_type=Exception)
            def func():
                raise Exception

            @contextmanager
            def my_context(self, e, i, **kwargs):
                ...

            @retry.add_try(err_context=my_context)
            def func():
                raise Exception
    """

    def __init__(self, verbose=True, stdout_method=print, msg_fmt=None, count=3, wait=15):
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.count = count
        self.wait = wait
        self.msg_fmt = msg_fmt or 'Something error occur: "{e}", sleep {wait} seconds, and then retry!'

    @contextmanager
    def context(self, *args, e=None, i=None, **kwargs):
        msg = self.msg_fmt.format(e=e, wait=self.wait, **kwargs)
        self.stdout_method(msg)
        try:
            yield
        finally:
            self.stdout_method(f'{i + 2}th process!')

    def add_try(
            self,
            err_context=None,
            err_type=(ConnectionError, TimeoutError),
            raise_type=type(None),
    ):
        context = err_context or self.context

        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                for i in range(self.count):
                    try:
                        return func(*args, **kwargs)

                    except err_type as e:
                        if isinstance(e, raise_type):
                            raise e

                        if i >= self.count - 1:
                            raise e

                        with context(*args, e=e, i=i, **kwargs):
                            time.sleep(self.wait)

            return wrap

        return wrap2


retry = Retry()


class RegisterTables:
    """
    Usage:
        .. code-block:: python

            register_tables = RegisterTables()

            @register_tables.add_register()
            class SimpleClass:
                ...

            cls = register_tables.get('SimpleClass')

            @register_tables.add_register('k1', 't1')
            class SimpleClass:
                ...

            cls = register_tables.get('k1', 't1')
    """

    def add_register(self, key='', table_name='default'):
        def wrap(func):
            if not hasattr(self, table_name):
                setattr(self, table_name, {})

            getattr(self, table_name)[key or func.__name__] = func

            return func

        return wrap

    def get(self, key, table_name='default'):
        return getattr(self, table_name)[key]

    def __repr__(self):
        return str(self.__dict__)


register_tables = RegisterTables()
