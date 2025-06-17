"""utils for operating the tritonclient"""
import json
import re
from typing import Optional, Any

import numpy as np
import requests
import tritonclient  # pip install tritonclient[all]
import tritonclient.grpc
import tritonclient.http
from tritonclient.utils import InferenceServerException

from . import converter, log_utils
from functools import partial

# https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes
datatypes = {
    'Config': ['TYPE_BOOL', 'TYPE_UINT8', 'TYPE_UINT16', 'TYPE_UINT32', 'TYPE_UINT64', 'TYPE_INT8', 'TYPE_INT16', 'TYPE_INT32', 'TYPE_INT64', 'TYPE_FP16', 'TYPE_FP32', 'TYPE_FP64', 'TYPE_STRING', 'TYPE_BF16'],
    'API': ['BOOL', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'INT8', 'INT16', 'INT32', 'INT64', 'FP16', 'FP32', 'FP64', 'BYTES', 'BF16'],
    'TensorRT': ['kBOOL', 'kUINT8', '', '', '', 'kINT8', '', 'kINT32', '', 'kHALF', 'kFLOAT', '', '', ''],
    'TensorFlow': ['DT_BOOL', 'DT_UINT8', 'DT_UINT16', 'DT_UINT32', 'DT_UINT64', 'DT_INT8', 'DT_INT16', 'DT_INT32', 'DT_INT64', 'DT_HALF', 'DT_FLOAT', 'DT_DOUBLE', 'DT_STRING', ''],
    'ONNX': ['BOOL', 'UINT8', 'UINT16', 'UINT32', 'UINT64', 'INT8', 'INT16', 'INT32', 'INT64', 'FLOAT16', 'FLOAT', 'DOUBLE', 'STRING', ''],
    'PyTorch': ['kBool', 'kByte', '', '', '', 'kChar', 'kShort', 'kInt', 'kLong', '', 'kFloat', 'kDouble', '', ''],
    'NumPy': ['BOOL', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'BYTES', '']
}


class BaseClient:
    def __init__(
            self,
            client: Optional[tritonclient.http.InferenceServerClient | tritonclient.grpc.InferenceServerClient],
            verbose=False, logger=None, **kwargs
    ):
        self.client = client
        self.verbose = verbose
        self.logger = log_utils.get_logger(logger)
        self.model_configs = {}
        self.model_versions = {}

    def init(self):
        raise NotImplementedError()

    def _init(self, model_info):
        for info in model_info:
            name = info['name']
            if 'version' not in info:
                continue
            version = info['version']
            state = info['state']

            if state != 'READY':
                self.logger.warning(f'{name}:{version}({state}) is not ready, pls check!')
                continue

            self.model_versions[name] = version
            if isinstance(self.client, tritonclient.http.InferenceServerClient):
                self.model_configs[name, version] = self.client.get_model_config(name, version)
            else:
                self.model_configs[name, version] = self.client.get_model_config(name, version, as_json=True)['config']

        if self.verbose:
            self.logger.info(self.model_configs)

    def load(self, model_name):
        """must work on explicit model
        run tritonserver with `--model-control-mode explicit`"""
        self.client.load_model(model_name)
        self.init()

    def unload(self, model_name):
        self.client.unload_model(model_name)
        self.init()

    def _get_model_configs(self, model_name, model_version=None):
        model_version = model_version or self.model_versions.get(model_name)

        # firstly, try to reinit the model_configs
        if (model_name, model_version) not in self.model_configs:
            self.logger.warning(f'Can not get model configs for {model_name = } and {model_version = }, try to reinit the model_configs!')
            self.init()

            model_version = model_version or self.model_versions.get(model_name)

            # secondly, try to load the model, and then reinit the model_configs
            if (model_name, model_version) not in self.model_configs:
                self.logger.warning(f'Can not get model configs for {model_name = } and {model_version = }, try to reload the model!')
                self.load(model_name)
                self.init()

                model_version = model_version or self.model_versions.get(model_name)

                # finally, raise the error
                assert (model_name, model_version) in self.model_configs, \
                    f'Got {model_name = } and {model_version = }, where the keys is {self.model_configs.keys()}, pls check'

        model_config = self.model_configs[model_name, model_version]

        return model_version, model_config

    def _parse_input(self, i, cfg):
        dtype = cfg['data_type']
        if dtype == 'TYPE_STRING':
            if isinstance(i, list):
                i = np.array([_.encode('utf8') for _ in i], dtype=np.object_)
            else:
                i = np.array([i.encode('utf8')], dtype=np.object_)

        elif dtype == 'TYPE_BOOL':
            if isinstance(i, list):
                i = np.array(i)
            else:
                i = np.array([i])

        elif isinstance(i, list):
            i = np.array(i)

        dtype = datatypes['API'][datatypes['Config'].index(dtype)]
        return i, dtype

    def _parse_output(self, result):
        if isinstance(result, tritonclient.http.InferResult):
            response = result.get_response()
        else:
            response = result.get_response(as_json=True)

        outputs = {}
        for output in response['outputs']:
            name = output['name']
            datatype = output['datatype']
            o = result.as_numpy(name)
            if datatype == 'BYTES':
                if len(o.shape) == 0:
                    o = o.decode('utf-8')
                else:
                    o = [_.decode('utf-8') for _ in o]
            outputs[name] = o

        return outputs


class HttpClient(BaseClient):
    """
    note, InferenceServerClient intended to be used by a single thread,
    if using in a multi-thread like flask app, usually got errors like
    `greenlet.error: cannot switch to a different thread (which happens to have exited)`
    so using the following scripts to initialize the client to avoid the errors:
        ```
        class App:
            @property
            def trt_client():
                return HttpClient()
        ```
    instead of:
        ```
        class App:
            def __init__():
                self.trt_client = HttpClient()
        ```

    """

    def __init__(self, url='127.0.0.1:8000', verbose=False, **kwargs):
        self.url = url

        client = tritonclient.http.InferenceServerClient(url=url, verbose=False, **kwargs)
        super().__init__(client, verbose=verbose, **kwargs)

    def init(self):
        model_info = self.client.get_model_repository_index()
        self._init(model_info)

    def async_infer(self, *inputs: Any, model_name, model_version=None):
        model_version, model_config = self._get_model_configs(model_name, model_version)

        _inputs = []
        for cfg, i in zip(model_config['input'], inputs):
            i, dtype = self._parse_input(i, cfg)

            _input = tritonclient.http.InferInput(cfg['name'], i.shape, dtype)
            _input.set_data_from_numpy(i)
            _inputs.append(_input)

        _outputs = []
        for cfg in model_config['output']:
            _output = tritonclient.http.InferRequestedOutput(cfg['name'])
            _outputs.append(_output)

        async_req = self.client.async_infer(
            model_name=model_name,
            model_version=model_version,
            inputs=_inputs,
            outputs=_outputs
        )

        return async_req

    def async_get(self, async_req: tritonclient.http.InferAsyncRequest):
        result = async_req.get_result()
        outputs = self._parse_output(result)

        return outputs

    def generate(self, *inputs: Any, model_name, model_version=None):
        model_version, model_config = self._get_model_configs(model_name, model_version)

        if model_version:
            model_version = f'/versions/{model_version}'
        else:
            model_version = ''

        url = f'http://{self.url}/v2/models/{model_name}{model_version}/generate'

        _inputs = {}
        for cfg, i in zip(model_config['input'], inputs):
            _inputs[cfg['name']] = converter.DataConvert.custom_to_constant(i)

        _inputs = json.dumps(_inputs, ensure_ascii=False)
        r = requests.post(url, data=_inputs)
        if r.status_code != 200:
            raise InferenceServerException(r.text)
        result = r.json()
        outputs = {}
        for output_config in model_config['output']:
            name = output_config['name']
            outputs[name] = result[name]
        return outputs

    def generate_stream(self, *inputs: Any, model_name, model_version=None):
        """official http triton client do not support generate endpoint called
        see https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2450/user-guide/docs/protocol/extension_generate.html"""
        model_version, model_config = self._get_model_configs(model_name, model_version)

        if model_version:
            model_version = f'/versions/{model_version}'
        else:
            model_version = ''

        url = f'http://{self.url}/v2/models/{model_name}{model_version}/generate_stream'

        _inputs = {}
        for cfg, i in zip(model_config['input'], inputs):
            _inputs[cfg['name']] = converter.DataConvert.custom_to_constant(i)

        _inputs = json.dumps(_inputs, ensure_ascii=False)
        r = requests.post(url, data=_inputs, stream=True)

        for chunk in r.iter_lines():
            if chunk:
                data = chunk.decode('utf8')
                r = re.search(r'^data: (.*)$', data)
                if r:
                    data = r.group(1)
                    result = json.loads(data)
                    outputs = {}
                    for output_config in model_config['output']:
                        name = output_config['name']
                        outputs[name] = result[name]
                    yield outputs


class GrpcClient(BaseClient):
    def __init__(self, url='127.0.0.1:8001', verbose=False, **kwargs):
        self.url = url

        client = tritonclient.grpc.InferenceServerClient(url=url, verbose=False, **kwargs)
        super().__init__(client, verbose=verbose, **kwargs)

    def init(self):
        model_info = self.client.get_model_repository_index(as_json=True)['models']
        self._init(model_info)

    def generate(self, *inputs: Any, model_name, model_version=None, callback=None):
        model_version, model_config = self._get_model_configs(model_name, model_version)

        _inputs = []
        for cfg, i in zip(model_config['input'], inputs):
            i, dtype = self._parse_input(i, cfg)

            _input = tritonclient.grpc.InferInput(cfg['name'], i.shape, dtype)
            _input.set_data_from_numpy(i)
            _inputs.append(_input)

        _outputs = []
        for cfg in model_config['output']:
            _output = tritonclient.grpc.InferRequestedOutput(cfg['name'])
            _outputs.append(_output)

        total_outputs = []

        self.client.start_stream(partial(callback or self.stream_callback, total_outputs=total_outputs))
        self.client.async_stream_infer(
            model_name=model_name,
            model_version=model_version,
            inputs=_inputs,
            outputs=_outputs
        )

        self.client.stop_stream()
        outputs = {}
        for output_config in model_config['output']:
            name = output_config['name']
            output = [a for o in total_outputs for a in o[name]]
            if output_config['data_type'] == 'TYPE_STRING':
                output = ''.join(output)
            outputs[name] = output
        return outputs

    def stream_callback(self, result: tritonclient.grpc.InferResult, error, total_outputs=[]):
        if error is not None:
            raise Exception(error)
        outputs = self._parse_output(result)
        total_outputs.append(outputs)

    def generate_stream(self, *inputs: Any, model_name, model_version=None, callback=None):
        raise NotImplementedError


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    this is an example
    refer to: https://github.com/triton-inference-server/python_backend
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Args:
            args:
              Both keys and values are strings. The dictionary keys and values are:
              * model_config: A JSON string containing the model configuration
              * model_instance_kind: A string containing model instance kind
              * model_instance_device_id: A string containing model instance device ID
              * model_repository: Model repository path
              * model_version: Model version
              * model_name: Model name
        """
        import triton_python_backend_utils as pb_utils
        import json

        # get configs
        # configs can be found in `config.pbtxt`
        # args would like that
        # args = {
        #     'model_config': {
        #         "input": [{"name": "", "data_type": "", "format": "", "dims": []}],
        #         "output": [{"name": "", "data_type": "", "dims": [], }],
        #         "instance_group": [{"name": "", "kind": "KIND_GPU", "gpus": []}],
        #     },
        #     'model_instance_kind': '',
        #     'model_instance_name': '',
        #     'model_instance_device_id': '',
        #     'model_repository': '',
        #     'model_version': '',
        #     'model_name': ''
        # }
        self.model_config = json.loads(args['model_config'])

        self.input_names = [i['name'] for i in self.model_config['input']]
        self.input_dtypes = [pb_utils.triton_string_to_numpy(i['data_type']) for i in self.model_config['input']]
        self.output_names = [i['name'] for i in self.model_config['output']]
        self.output_dtypes = [pb_utils.triton_string_to_numpy(i['data_type']) for i in self.model_config['output']]

        self.device_id = args['model_instance_device_id']
        ...

        # init model
        self.model = ...

    def execute(self, requests):
        import triton_python_backend_utils as pb_utils

        responses = []
        for request in requests:
            # get inputs
            in_0 = pb_utils.get_input_tensor_by_name(request, self.input_names[0])
            ...

            # get outputs from model inference
            out_0, *outs = self.model(in_0, ...)

            out_tensor_0 = pb_utils.Tensor(self.output_names[0], out_0.astype(self.output_dtypes[0]))
            ...

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, ...])

            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pass
