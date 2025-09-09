"""utils for operating instances of python, decorator usually"""
import time
from functools import wraps

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

    def __init__(self, verbose=True, stdout_method=print):
        self.stdout_method = stdout_method if verbose else FakeIo()

    def add_ignore(
            self,
            error_message='',
            err_type=(ConnectionError, TimeoutError)
    ):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                try:
                    return func(*args, **kwargs)

                except err_type as e:
                    msg = error_message or f'Something error occur: {e}'
                    self.stdout_method(msg)

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

            @retry.add_try(error_message='there is an error, sleep %d seconds')
            def func():
                raise Exception

            @retry.add_try(err_type=Exception)
            def func():
                raise Exception
    """

    def __init__(self, verbose=True, stdout_method=print, count=3, wait=15):
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.count = count
        self.wait = wait

    def add_try(
            self,
            error_message='',
            err_type=(ConnectionError, TimeoutError)
    ):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                for i in range(self.count):
                    try:
                        return func(*args, **kwargs)

                    except err_type as e:
                        if i >= self.count - 1:
                            raise e

                        msg = error_message or 'Something error occur: "{e}", sleep {wait} seconds, and then retry!'
                        msg = msg.format(e=e, wait=self.wait)
                        self.stdout_method(msg)
                        time.sleep(self.wait)
                        self.stdout_method(f'{i + 2}th try!')

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

    def get(self, key, table_name='default', default=None):
        return getattr(self, table_name).get(key, default)

    def __repr__(self):
        return str(self.__dict__)


register_tables = RegisterTables()
