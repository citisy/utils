"""utils for creating a web app to provide api endpoints"""
import json
import os
import time
from contextlib import nullcontext

import pydantic

from . import converter


class BaseApp:
    @classmethod
    def from_configs(cls, configs: dict, app_configs=dict()):
        """
        configs:
            {router_path: {api_path: router_kwargs}}
        router_kwargs:
            app_func
            method
            func
            request_template
            response_template
            func_configs
        """
        app = cls.create_app(**app_configs)

        for router_path, cfg in configs.items():
            sub_app = cls.create_sub_app()
            for api_path, router_kwargs in cfg.items():
                if 'app_func' in router_kwargs:
                    app_func = router_kwargs.get('app_func')
                    app_func = converter.DataInsConvert.str_to_instance(app_func)
                    app_func(sub_app, router_path, api_path, **router_kwargs)
                else:
                    method = router_kwargs.get('method', 'post').lower()
                    if method == 'post':
                        cls.register_post_router(sub_app, router_path, api_path, **router_kwargs)
                    elif method == 'get':
                        cls.register_get_router(sub_app, router_path, api_path, **router_kwargs)
                    else:
                        raise NotImplementedError(f"method {method} not supported")

            cls.mount_app(app, sub_app, router_path)

        app = cls.wrap_app(app)

        return app

    @classmethod
    def create_app(cls, **app_configs):
        raise NotImplementedError

    @classmethod
    def create_sub_app(cls):
        raise NotImplementedError

    @classmethod
    def register_post_router(cls, app, router_path, api_path, **kwargs):
        raise NotImplemented

    @classmethod
    def register_get_router(cls, app, router_path, api_path, **kwargs):
        raise NotImplemented

    @classmethod
    def mount_app(cls, app, sub_app, router_path, **kwargs):
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
            router_path,
            api_path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        request_template = dict if request_template is None else request_template
        response_template = None if response_template is None else response_template

        @app.post(api_path, response_model=response_template, **method_configs)
        def post(data: request_template):
            if isinstance(data, pydantic.BaseModel):
                data = data.dict(exclude_none=True)
            ret = func(data, **func_configs)
            return ret

    @staticmethod
    def register_get_router(
            app: 'FastAPI' or 'APIRouter',
            router_path,
            api_path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        request_template = dict if request_template is None else request_template
        response_template = None if response_template is None else response_template

        @app.get(api_path, response_model=response_template, **method_configs)
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

    @classmethod
    def mount_app(cls, app, sub_app, router_path, **kwargs):
        app.include_router(sub_app, prefix=router_path)


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
            router_path,
            api_path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        from flask import jsonify, request

        @app.post(api_path, endpoint=api_path, **method_configs)
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
            router_path,
            api_path,
            func=None,
            request_template: 'pydantic.BaseModel()' = None,
            response_template: 'pydantic.BaseModel()' = None,
            func_configs: dict = {},
            method_configs: dict = {},
            **ignore_kwargs
    ):
        from flask import jsonify, request

        @app.get(api_path, endpoint=api_path, **method_configs)
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


class FastApiAppWrapper:
    @staticmethod
    def add_CORS(app):
        from fastapi.middleware.cors import CORSMiddleware

        # 添加跨域中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        return app

    @staticmethod
    def add_process_time_header(app):
        @app.middleware("http")
        async def add(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = f'{process_time:0.4f} sec'
            return response

    @staticmethod
    def add_docs(
            app, router_path, api_path,
            title='',
            version="1.0.0",
            description=None,
            **router_kwargs
    ):
        from fastapi.openapi.docs import get_swagger_ui_html
        from fastapi import HTTPException
        from fastapi.openapi.utils import get_openapi
        from fastapi.responses import FileResponse

        @app.get(f"/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url=f'{router_path}/openapi.json',
                title=title + " - Swagger UI",
                swagger_js_url=f"{router_path}/static/swagger-ui-bundle.js",
                swagger_css_url=f"{router_path}/static/swagger-ui.css",
                swagger_favicon_url=f"{router_path}/static/favicon-32x32.png",
            )

        @app.get("/static/{file_path:path}", include_in_schema=False)
        async def get_static_file(file_path: str):
            file_location = os.path.join('static', file_path)

            if not os.path.exists(file_location) or not os.path.isfile(file_location):
                raise HTTPException(status_code=404, detail="File not found")

            return FileResponse(file_location)

        @app.get("/openapi.json", include_in_schema=False)
        async def custom_openapi():
            openapi_schema = get_openapi(
                title=title,
                version=version,
                routes=app.routes,
                description=description
            )
            openapi_schema['paths'] = {router_path + k: v for k, v in openapi_schema['paths'].items()}
            return openapi_schema
