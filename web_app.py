"""utils for creating a web app to provide api endpoints"""
import json
from contextlib import nullcontext

import pydantic

from . import converter


class BaseApp:
    @classmethod
    def from_configs(cls, configs: dict, app_configs=dict()):
        """
        configs:
            {path1: {path2: router_kwargs}}
        router_kwargs:
            app_func
            method
            func
            request_template
            response_template
            func_configs
        """
        app = cls.create_app(**app_configs)

        for path1, cfg in configs.items():
            sub_app = cls.create_sub_app()
            for path2, router_kwargs in cfg.items():
                if 'app_func' in router_kwargs:
                    app_func = router_kwargs.get('app_func')
                    app_func = converter.DataInsConvert.str_to_instance(app_func)
                    app_func(sub_app, path2, **router_kwargs)
                else:
                    method = router_kwargs.get('method', 'post').lower()
                    if method == 'post':
                        cls.register_post_router(sub_app, path2, **router_kwargs)
                    elif method == 'get':
                        cls.register_get_router(sub_app, path2, **router_kwargs)
                    else:
                        raise NotImplementedError(f"method {method} not supported")

            app.include_router(sub_app, prefix=path1)

        app = cls.wrap_app(app)

        return app

    @classmethod
    def create_app(cls, **app_configs):
        raise NotImplementedError

    @classmethod
    def create_sub_app(cls):
        raise NotImplementedError

    @classmethod
    def register_post_router(cls, app, path, **kwargs):
        raise NotImplemented

    @classmethod
    def register_get_router(cls, app, path, **kwargs):
        raise NotImplemented

    @classmethod
    def wrap_app(cls, app, **kwargs):
        return app


class FastapiOp(BaseApp):
    """
    op = FastapiOp
    app = op.create_app()
    op.register_post_router(app, path='/test', model=model)

    if __name__ == '__main__':
        import uvicorn
        uvicorn.run(app)
    """

    @staticmethod
    def create_app(**app_configs):
        from fastapi import FastAPI

        return FastAPI(**app_configs)

    @staticmethod
    def create_sub_app():
        from fastapi import APIRouter

        return APIRouter()

    @staticmethod
    def register_post_router(
            app: 'FastAPI' or 'APIRouter',
            path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        request_template = dict if request_template is None else request_template
        response_template = None if response_template is None else response_template

        @app.post(path, response_model=response_template, **method_configs)
        def post(data: request_template):
            if isinstance(data, pydantic.BaseModel):
                data = data.dict(exclude_none=True)
            ret = func(data, **func_configs)
            return ret

    @staticmethod
    def register_get_router(
            app: 'FastAPI' or 'APIRouter',
            path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        request_template = dict if request_template is None else request_template
        response_template = None if response_template is None else response_template

        @app.get(path, response_model=response_template, **method_configs)
        def get(data: request_template):
            if isinstance(data, pydantic.BaseModel):
                data = data.dict()
            ret = func(data, **func_configs)
            return ret

    @staticmethod
    def wrap_app(app, **kwargs):
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return app


class FlaskOp(BaseApp):
    """
    op = FastapiOp
    app = op.create_app()
    op.register_post_router(app, path='/test', model=model)

    if __name__ == '__main__':
        app.run()
    """

    @staticmethod
    def create_app():
        from flask import Flask

        return Flask(__name__)

    @staticmethod
    def create_sub_app(name):
        from flask import Blueprint

        return Blueprint(name, __name__)

    @staticmethod
    def register_post_router(
            app: 'Flask' or 'Blueprint',
            path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        from flask import jsonify, request

        @app.post(path, endpoint=path, **method_configs)
        def post():
            data = request.get_data().decode('utf-8')
            data = json.loads(data)
            if request_template:
                data = request_template(**data)
                data = data.dict(exclude_none=True)

            ret = func(data, **func_configs)

            if response_template:
                ret = response_template(**ret)
                ret = ret.dict()

            return jsonify(ret)

    @staticmethod
    def register_get_router(
            app: 'Flask' or 'Blueprint',
            path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        from flask import jsonify, request

        @app.get(path, endpoint=path, **method_configs)
        def get():
            data = request.args.to_dict()
            if request_template:
                data = request_template(**data)
                data = data.dict(exclude_none=True)

            ret = func(data, **func_configs)

            if response_template:
                ret = response_template(**ret)
                ret = ret.dict()

            return jsonify(ret)


class FakeApp:
    """a placeholder, empty endpoint method to cheat some functions which must use an endpoint method,
    it means that the method do nothing in fact,
    it is useful to reduce the number of code changes

    Examples
    .. code-block:: python

        # real app
        app = FastAPI()

        # fake app
        app = FakeApp()

        @app.route(...)
        def func(...):
            ...
    """

    def __init__(self, *args, **kwargs):
        self.config = dict()
        self.conf = dict()
        self.__dict__.update(kwargs)

    def register_blueprint(self, *args, **kwargs):
        pass

    def route(self, *args, **kwargs):
        return nullcontext

    def post(self, *args, **kwargs):
        return nullcontext

    def get(self, *args, **kwargs):
        return nullcontext
