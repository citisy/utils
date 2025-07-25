"""utils for providing helper with torch, easy to log, manage your torch modules"""
import copy
import math
import os
import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import math_utils


def setup_seed(seed=42):
    """42 is a lucky number"""
    import random
    import torch.backends.cudnn as cudnn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class ModuleInfo:
    @classmethod
    def std_profile(cls, model, depth=None, human_readable=True):
        from .visualize import TextVisualize

        profile = cls.profile_per_layer(model, depth=depth)
        cols = ('name', 'module', 'params', 'grads', 'args')
        lens = [-1] * len(cols)
        infos = []
        for p in profile:
            info = (
                p[0],
                p[1],
                TextVisualize.num_to_human_readable_str(p[2]["params"]) if human_readable else p[2]["params"],
                TextVisualize.num_to_human_readable_str(p[2]["grads"]) if human_readable else p[2]["grads"],
                TextVisualize.dict_to_str(p[2]["args"])
            )
            infos.append(info)
            for i, s in enumerate(info):
                l = len(str(s))
                if lens[i] < l:
                    lens[i] = l

        template = ''
        for l in lens:
            template += f'%-{l + 3}s'

        s = 'module info: \n'
        s += template % cols + '\n'
        s += template % tuple('-' * l for l in lens) + '\n'

        for info in infos:
            s += template % info + '\n'

        params = sum([p[2]["params"] for p in profile])
        grads = sum([p[2]["grads"] for p in profile])
        if human_readable:
            params = TextVisualize.num_to_human_readable_str(params)
            grads = TextVisualize.num_to_human_readable_str(grads)

        s += template % tuple('-' * l for l in lens) + '\n'
        s += template % ('sum', '', params, grads, '')
        return s, infos

    @classmethod
    def profile_per_layer(cls, module: nn.Module, depth=None):
        profiles = []

        def cur(current_m, dep, prev_name=''):
            for name, m in current_m._modules.items():
                if m is None:
                    continue
                if dep <= 1 or len(m._modules) == 0:
                    name = f'{prev_name}.{name}'[1:]
                    profiles.append((name, str(type(m))[8:-2], cls.profile(m)))
                else:
                    cur(m, dep - 1, f'{prev_name}.{name}')

        depth = depth or float('inf')
        cur(module, depth)
        return profiles

    @classmethod
    def profile(cls, module):
        return dict(
            params=cls.profile_params(module),
            grads=cls.profile_grads(module),
            args=cls.profile_args(module)
        )

    @staticmethod
    def profile_params(module):
        return sum(x.numel() for x in module.parameters() if not isinstance(x, nn.UninitializedParameter))

    @staticmethod
    def profile_grads(module):
        return sum(x.numel() for x in module.parameters() if x.requires_grad and not isinstance(x, nn.UninitializedParameter))

    @staticmethod
    def profile_args(module):
        args = {}
        if hasattr(module, 'in_channels'):
            args['i_ch'] = module.in_channels
        if hasattr(module, 'out_channels'):
            args['o_ch'] = module.out_channels
        if hasattr(module, 'in_features'):
            args['i_f'] = module.in_features
        if hasattr(module, 'out_features'):
            args['o_f'] = module.out_features
        if hasattr(module, 'input_size'):
            args['i_size'] = module.input_size
        if hasattr(module, 'output_size'):
            args['o_size'] = module.output_size
        if hasattr(module, 'num_embeddings'):
            args['n_emb'] = module.num_embeddings
        if hasattr(module, 'embedding_dim'):
            args['emb_dim'] = module.embedding_dim
        if hasattr(module, 'kernel_size'):
            k = module.kernel_size
            if isinstance(k, (list, tuple)):
                k = k[0]
            args['k'] = k
        if hasattr(module, 'stride'):
            s = module.stride
            if isinstance(s, (list, tuple)):
                s = s[0]
            args['s'] = s
        if hasattr(module, 'padding'):
            p = module.padding
            if isinstance(p, (list, tuple)):
                p = p[0]
            args['p'] = p

        return args

    @staticmethod
    def profile_flops(module, input_size, *test_args):
        import thop

        p = next(module.parameters())
        test_data = torch.empty((1, *input_size), device=p.device)
        flops = thop.profile(copy.deepcopy(module), inputs=(test_data, *test_args), verbose=False)[0] / 1E9 * 2  # stride GFLOPs

        return flops

    @staticmethod
    def possible_device(module):
        """Returns the first found device in parameters, otherwise returns the first found device in tensors."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = module._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @staticmethod
    def possible_dtype(module):
        """Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found."""
        last_dtype = None
        for t in module.parameters():
            last_dtype = t.dtype
            if t.is_floating_point():
                return t.dtype

        if last_dtype is not None:
            # if no floating dtype was found return whatever the first dtype is
            return last_dtype

        else:
            # For nn.DataParallel compatibility in PyTorch > 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = module._named_members(get_members_fn=find_tensor_attributes)
            last_tuple = None
            for tuple in gen:
                last_tuple = tuple
                if tuple[1].is_floating_point():
                    return tuple[1].dtype

            # fallback to the last dtype
            return last_tuple[1].dtype


class ModuleManager:
    @staticmethod
    def is_parallel(module):
        """Returns True if model is of type DP or DDP"""
        return type(module) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    @classmethod
    def de_parallel(cls, module):
        """De-parallelize a model: returns single-GPU model if model is of type DP or DDP"""
        return module.module if cls.is_parallel(module) else module

    @staticmethod
    def torch_gc():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except AssertionError as e:
            print(e)

    @staticmethod
    def freeze_module(module: nn.Module, allow_train=False):
        module.requires_grad_(False)
        if not allow_train:
            # module only be allowed to eval, does not change to train mode anymore
            module.eval()
            module.train = lambda self, mode=True: self

    @staticmethod
    def quantized_by_pytorch(module: nn.Module, trace_func=None, backend='fbgemm'):
        """see https://pytorch.org/docs/stable/quantization.html"""
        module.eval()
        module.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(module, inplace=True)
        if trace_func is not None:
            with torch.no_grad():
                # help to collect the running info for quantization
                trace_func(module)
        torch.quantization.convert(module, inplace=True)

    @staticmethod
    def quantized_by_quanto(module: nn.Module, weights='qint8', activations=None, **kwargs):
        """see https://github.com/huggingface/quanto"""
        import quanto
        quanto.quantize(module, weights=weights, activations=activations, **kwargs)
        for name, m in module.named_modules():
            if hasattr(m, 'freeze'):
                m.freeze()

    @classmethod
    def low_memory_run(cls, module: nn.Module, call_func, device, *args, require_grad=True, **kwargs):
        """only send the module to gpu when the module need to be run,
        and the gpu will be released after running"""
        module.to(device)
        obj = call_func(*args, **kwargs)

        if not (require_grad and module.training):
            module.cpu()
            cls.torch_gc()

        return obj

    @staticmethod
    def assign_device_run(module: nn.Module, call_func, device, *args, force_effect_module=True, **kwargs):
        """let module run in the assigned device"""
        if force_effect_module:
            module.to(device)

        args = [obj.to(device) if isinstance(obj, torch.Tensor) else obj for obj in args]
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        return call_func(*args, **kwargs)

    @staticmethod
    def assign_dtype_run(module: nn.Module, call_func, dtype, *args, force_effect_module=True, **kwargs):
        """let module run in the assigned dtype"""
        if force_effect_module:
            module.to(dtype)

        check = lambda obj: isinstance(obj, torch.Tensor) and obj.dtype.is_floating_point == dtype.is_floating_point
        args = [obj.to(dtype) if check(obj) else obj for obj in args]
        kwargs = {k: v.to(dtype) if check(v) else v for k, v in kwargs.items()}

        return call_func(*args, **kwargs)

    @staticmethod
    def single_batch_run(module: nn.Module, call_func, *args, **kwargs):
        """let module run one after another single batch"""
        check = lambda obj: isinstance(obj, (torch.Tensor, list, tuple))
        b = None
        for obj in args:
            if check(obj):
                b = len(obj)
                break

        temp = []
        for i in range(b):
            tmp_args = [obj[i:i + 1] if check(obj) else obj for obj in args]
            tmp_kwargs = {k: obj[i:i + 1] if check(obj) else obj for k, obj in kwargs.items()}
            rets = call_func(*tmp_args, **tmp_kwargs)
            temp.append(rets)

        if isinstance(temp[0], torch.Tensor):
            temp = torch.cat(temp)
        else:
            temp = tuple((torch.cat(t) for t in math_utils.transpose(temp)))
        return temp

    @staticmethod
    def checkpoint(module: nn.Module, call_func, *args, is_first_layer=False, **kwargs):
        """note, if using checkpoint, it is best not to use it in the first layer of the module,
        as it usually does not contain gradients, thought can set `x.requires_grad_(True)` to pass it,
        but it does not work yet always"""
        from torch.utils.checkpoint import checkpoint

        if module.training:  # only work on train step
            # prevent to no grad
            if is_first_layer:
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:  # only float tensor has grad
                        arg.requires_grad_(True)
                for arg in kwargs.values():
                    if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:  # only float tensor has grad
                        arg.requires_grad_(True)
            # note, if having kwargs, use `use_reentrant=False`
            return checkpoint(call_func, *args, use_reentrant=False, **kwargs)
        else:
            return call_func(*args, **kwargs)

    @classmethod
    def initialize_layers(cls, module, init_gain=0.02, init_type='normal'):
        """trace each module, initialize the variables
        if module has `initialize_layers`, use `module.initialize_layers()` to initialize"""

        def cur(current_m):
            for name, m in current_m._modules.items():
                if m is None:
                    continue

                if hasattr(m, 'initialize_layers'):
                    m.initialize_layers()
                    continue

                t = type(m)

                if t is nn.BatchNorm2d:
                    # m.eps = 1e-3
                    # m.momentum = 0.03
                    m.weight.data.normal_(1.0, init_gain)
                    m.bias.data.fill_(0.)

                elif t is nn.LayerNorm:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

                elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                    m.inplace = True

                elif t in [nn.Conv2d, nn.Linear, nn.Embedding]:
                    if init_type == 'normal':
                        nn.init.normal_(m.weight, 0.0, init_gain)
                    elif init_type == 'xavier':
                        nn.init.xavier_normal_(m.weight, gain=init_gain)
                    elif init_type == 'kaiming':
                        nn.init.kaiming_normal_(m.weight, a=0)
                    elif init_type == 'orthogonal':
                        nn.init.orthogonal_(m.weight, gain=init_gain)

                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

                elif t in [nn.ConvTranspose2d]:
                    m.weight.data.copy_(cls.bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))

                if len(m._modules) != 0:
                    cur(m)

        cur(module)

    @staticmethod
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
        f = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        ch = min(in_channels, out_channels)
        weight[range(ch), range(ch), :, :] = f
        return weight

    @staticmethod
    def get_module_by_name(module, tensor_name: str):
        if "." in tensor_name:
            splits = tensor_name.split(".")
            for split in splits:
                new_module = getattr(module, split)
                if new_module is None:
                    raise ValueError(f"{module} has no attribute {split}.")
                module = new_module

        else:
            module = getattr(module, tensor_name)

        return module

    @staticmethod
    def get_module_by_key(module, key=None, include=(), exclude=(), is_last_module=False, is_return_last_module=False):
        """

        Args:
            module:
            key (str or nn.Module):
                same as key of 'include'
            include (List[str or nn.Module]):
            exclude (List[str or nn.Module]):
                if include is empty, find all the module not in exclude set
                if include is not empty, find all the module in include set, and then filter the module in the exclude set
            is_last_module:
                True to only check the last module
            is_return_last_module:
                True to return the module has found.
                Flase to return the parent of the module has found
        Returns:
            [[finded_module, name, full_name]]

        Examples:
            >>> ModuleManager.get_module_by_key(module, key='q')
            >>> ModuleManager.get_module_by_key(module, include=('q', 'k', 'v'), exclude=('l0.q', 'l0.k', 'l0.v'))

        """

        def cur(current_m: nn.Module, prev_name=''):
            for name, m in current_m._modules.items():
                if m is None:
                    continue

                full_name = f'{prev_name}.{name}'[1:]

                if is_last_module:
                    if is_find(full_name, m):
                        r.append((return_module(current_m, name), name, full_name))

                elif len(m._modules) == 0:
                    if is_find(full_name, m):
                        r.append((return_module(current_m, name), name, full_name))

                if len(m._modules) > 0:
                    cur(m, f'{prev_name}.{name}')

        def return_module(m, name=None):
            if is_return_last_module:
                return getattr(m, name)
            else:
                return m

        def is_find(name, m):
            flag = False
            for k in include:
                if is_last_module:
                    if (isinstance(k, str) and name.endswith(k)) or (not isinstance(k, str) and isinstance(m, k)):
                        flag = True

                else:
                    if (isinstance(k, str) and k in name) or (not isinstance(k, str) and isinstance(m, k)):
                        flag = True

            for k in exclude:
                if (isinstance(k, str) and k in name) or (not isinstance(k, str) and isinstance(m, k)):
                    flag = False
                elif not include:
                    flag = True

            return flag

        r = []
        if key is not None:
            include += (key,)
        cur(module)
        return r

    @classmethod
    def apply(cls, module, func, key=None, include=(), exclude=(), is_last_module=False, **func_kwargs):
        """
        Examples:
            # freeze encoder and decoder layer, train the head layer
            >>> ModuleManager.apply(nn.Module(), ModuleManager.freeze_module, include=('encoder', 'decoder'), exclude=('head', ), is_last_module=True)
        """
        objs = cls.get_module_by_key(module, key=key, include=include, exclude=exclude, is_last_module=is_last_module, is_return_last_module=True)

        for current_m, name, full_name in objs:
            func(current_m, **func_kwargs)


class WeightsFormats:
    formats = pd.DataFrame([
        ['PyTorch', '-', '.pt/.pth/.ckpt/.bin', True],
        ['TorchScript', 'torchscript', '.torchscript', True],
        ['Safetensors', 'safetensors', '.safetensors', True],
        ['ONNX', 'onnx', '.onnx', True],
        ['OpenVINO', 'openvino', '_openvino_model', False],
        ['TensorRT', 'engine', '.engine', True],
        ['CoreML', 'coreml', '.mlmodel', False],
        ['Keras', '-', '.h5', True],
        ['TensorFlow', '-', '.ckpt/.h5', True],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True],
        ['TensorFlow GraphDef', 'pb', '.pb', True],
        ['TensorFlow Lite', 'tflite', '.tflite', False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False],
        ['TensorFlow.js', 'tfjs', '_web_model', False],
        ['Paddle', '-', '.pdparams', True],
        ['Paddle Inference', '-', '.pdiparams/.pdmodel/.info', True],
    ], columns=['format', 'argument', 'suffix', 'GPU'])

    @classmethod
    def get_format_from_suffix(cls, save_path):
        suffix = Path(save_path).suffix
        k = None
        for i, row in cls.formats.iterrows():
            if suffix in row['suffix']:
                k = row['format']
                break

        return k


class Export:
    @staticmethod
    def to_safetensors(state_dict, f, **export_kwargs):
        from safetensors.torch import save_file

        save_file(state_dict, f, **export_kwargs)

    @staticmethod
    def to_torchscript(model, *trace_input, **export_kwargs):
        """note that, dynamic python script change to static c++ script, according to trace the code
        so, such as `if...else...`, 'for...in...`, etc., if trace in a dynamic variable,
        will cause some unexpectedly bugs
        recommend to export with cpu device"""
        model.eval()
        with torch.no_grad():
            # warmup, make sure that the model is initialized right
            model(*trace_input)
            jit_model = torch.jit.trace(model, trace_input, **export_kwargs)

        return jit_model

    @staticmethod
    def to_onnx(model, f, *trace_input, **export_kwargs):
        torch.onnx.export(model=model, f=f, args=trace_input, **export_kwargs)

    @staticmethod
    def to_state_dict_by_name(model, key=None, include=(), exclude=(), is_last=False):
        objs = ModuleManager.get_module_by_key(model, key=key, include=include, exclude=exclude, is_last_module=is_last)

        state_dict = OrderedDict()
        for current_m, _, full_name in objs:
            for name, tensors in current_m.state_dict():
                state_dict[f'{full_name}.{name}'] = tensors

        return state_dict


class Load:
    @classmethod
    def from_file(cls, save_path, **kwargs):
        load_dict = {
            'PyTorch': cls.from_ckpt,
            'TorchScript': cls.from_jit,
            'Safetensors': cls.from_safetensors,
            'Keras': cls.from_h5,
            'Paddle': cls.from_paddle
        }
        k = WeightsFormats.get_format_from_suffix(save_path)
        return load_dict.get(k)(save_path, **kwargs)

    @staticmethod
    def from_ckpt(save_path, **kwargs):
        return torch.load(save_path, **kwargs)

    @staticmethod
    def from_tf_ckpt(save_path, var_names=None, key_types=None, value_types=None, **kwargs):
        import tensorflow as tf

        assert var_names
        tensors = OrderedDict()
        for name in var_names:
            tensors[name] = torch.from_numpy(tf.train.load_variable(save_path, name))

        tensors = Converter.tensors_from_tf_to_torch(tensors, key_types, value_types)
        return tensors

    @staticmethod
    def from_h5(save_path, key_types=None, value_types=None, **kwargs):
        import h5py

        tensors = OrderedDict()
        with h5py.File(save_path, 'r') as f:
            if "layer_names" not in f.attrs and "model_weights" in f:
                f = f["model_weights"]
            layer_names = f.attrs['layer_names']

            for layer in layer_names:
                g = f[layer]
                weight_names = g.attrs['weight_names']
                for weight_name in weight_names:
                    tensors[weight_name.decode("utf8")] = torch.from_numpy(np.asarray(g[weight_name]))

        tensors = Converter.tensors_from_tf_to_torch(tensors, key_types, value_types)
        return tensors

    @staticmethod
    def from_safetensors(save_path, return_metadata=False, **kwargs):
        from safetensors import safe_open

        tensors = OrderedDict()
        with safe_open(save_path, framework="pt", **kwargs) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

            if return_metadata:
                tensors['__metadata__'] = f.metadata()

        return tensors

    @staticmethod
    def from_jit(save_path, **kwargs):
        return torch.jit.load(save_path, **kwargs)

    @staticmethod
    def from_paddle(save_path, **kwargs):
        import paddle  # see https://www.paddlepaddle.org.cn/install/quick
        tensors = paddle.load(save_path, **kwargs)
        tensors = {k: torch.from_numpy(v.numpy()) for k, v in tensors.items()}
        return tensors


class EarlyStopping:
    def __init__(self, thres=0.005, patience=None, min_period=0, ignore_min_score=-1,
                 verbose=True, stdout_method=print):
        """

        Args:
            thres: only updating the best_score and best_period when the increased score exceeds the threshold
            patience: early stop when accumulated period more than patience
            min_period: skip checking early stop when current period less than min_period
            ignore_min_score: skip checking early stop when current score less than ignore_min_score
            verbose: whether logging when early stop occurs or not
            stdout_method: logging method
        """
        self.thres = thres
        self.best_score = -1
        self.best_period = 0
        self.acc_period = 0
        self.last_period = 0
        self.min_period = min_period
        self.ignore_min_score = ignore_min_score
        self.patience = patience or float('inf')
        self.verbose = verbose
        self.stdout_method = stdout_method

    def step(self, period, score):
        if period < self.min_period or score < self.ignore_min_score:
            self.last_period = period
            self.best_score = score
            return False

        if score - self.best_score > self.thres:
            self.acc_period = 0
        elif abs(self.best_score - score) <= self.thres:
            self.acc_period += period - self.last_period

        if score > self.best_score:
            self.best_period = period
            self.best_score = score

        self.last_period = period
        stop = self.acc_period >= self.patience
        if stop and self.verbose:
            self.stdout_method(f'Early Stopping training. Best results observed at period {self.best_period}, and best score is {self.best_score}')
        return stop

    def __call__(self, period, score):
        return self.step(period, score)

    def state_dict(self):
        return dict(
            last_epoch=self.last_period,
            acc_epoch=self.acc_period,
            best_epoch=self.best_period,
            best_score=self.best_score
        )

    def load_state_dict(self, items: dict):
        self.__dict__.update(items)


class EMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, cur_step=0, step_start_ema=0, decay=0.9999, tau=2000):
        """
        as functions:
            - beta = decay * (1 - math.exp(-step / tau))
            - w = beta * w_o + (1- beta) * w_n
        there are some rows following:
            - step -> 0, beta -> 0, w -> w_n
            - step -> +inf, beta -> 1, w -> w_o
            - step -> 3*tau, beta -> exp(-3)*decay = 0.95*decay
              it gives that, w is w_o almost after 3*tau steps
            - tau -> +inf, beta -> 0
            - tau -> 0, beta -> decay
              it gives that, the ema model is update unavailable forever when tau=0 or tau=inf
              if tau=0, the ema model is the initializing model
              if tau=inf, the ema model is the training model
        """
        self.model = model
        self.ema_model = self.copy(model)
        self.cur_step = cur_step
        self.step_start_ema = step_start_ema
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))

    @staticmethod
    def copy(model):
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        return ema_model

    def restore(self):
        for c_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            param.data.copy_(c_param.data)

    @torch.no_grad()
    def update_attr(self):
        # larger the beta, more closed the weights to the old model
        beta = self.decay_fn(self.cur_step)

        msd = ModuleManager.de_parallel(self.model).state_dict()  # model state_dict
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v *= beta
                v += (1 - beta) * msd[k].to(v.device).detach()

    @torch.no_grad()
    def copy_attr(self, include=(), exclude=()):
        for k, v in self.model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema_model, k, v)
        self.ema_model.load_state_dict(self.model.state_dict())

    def step(self):
        if self.cur_step < self.step_start_ema:
            self.copy_attr()
        else:
            self.update_attr()
        self.cur_step += 1
        return self.cur_step

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.ema_model.load_state_dict(*args, **kwargs)


class Converter:
    @classmethod
    def force_to_tensors(cls, data, device):
        """Change object to gatherable in nn.DataParallel recursively

        The difference from to_device() is changing to torch.Tensor if float or int
        value is found.

        The restriction to the returned value in DataParallel:
            The object must be
            - torch.cuda.Tensor
            - 1 or more dimension. 0-dimension-tensor sends warning.
            or a list, tuple, dict.

        """
        if isinstance(data, dict):
            return {k: cls.force_to_tensors(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple, set)):
            if (
                    not len(data)
                    or not isinstance(data[0], (list, tuple, set))
                    or all(len(o) == len(data[0]) for o in data)
            ):
                try:
                    return torch.tensor(data).to(device)
                except:
                    return data
            else:
                return type(data)(cls.force_to_tensors(v, device) for v in data)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    @staticmethod
    def force_to_array(objs: Dict[Any, Optional[torch.Tensor]]) -> Dict[Any, np.ndarray]:
        for k, v in objs.items():
            if isinstance(k, torch.Tensor):
                if k.dtype in (torch.float16, torch.bfloat16, torch.float64):
                    v = v.to(torch.float32)
                objs[k] = v.cpu().numpy()

        return objs

    @staticmethod
    def convert_keys(state_dict: OrderedDict | dict, convert_dict: dict) -> OrderedDict:
        """
        Examples:
            >>> state_dict = {'before.a.1.weight': 1, 'before.b.2.weight': 2}

            >>> convert_dict = {'before': 'after'}
            >>> Converter.convert_keys(state_dict, convert_dict)
            OrderedDict([('after.a.1.weight', 1), ('after.b.2.weight', 2)])

            >>> convert_dict = {'before.{0}.': 'after.{0}.'}
            >>> Converter.convert_keys(state_dict, convert_dict)
            OrderedDict([('after.a.1.weight', 1), ('after.b.2.weight', 2)])

            >>> convert_dict = {'before.{0}.1': 'after.{0}.2'}
            >>> Converter.convert_keys(state_dict, convert_dict)
            OrderedDict([('after.a.2.weight', 1), ('before.b.2.weight', 2)])

            >>> convert_dict = {'before.{a}.{0}.': 'after.{0}.{a}.'}
            >>> Converter.convert_keys(state_dict, convert_dict)
            OrderedDict([('after.1.a.weight', 1), ('after.2.b.weight', 2)])

            >>> convert_dict = {'before.a.{0}.': 'after.a.{[0]+1}.'}
            >>> Converter.convert_keys(state_dict, convert_dict)
            OrderedDict([('after.a.2.weight', 1), ('before.b.2.weight', 2)])

            >>> convert_dict = {'before.{0}.': 'after.{"[0]".replace("a", "aa")}.'}
            >>> Converter.convert_keys(state_dict, convert_dict)
            OrderedDict([('after.aa.1.weight', 1), ('after.b.2.weight', 2)])

            >>> convert_dict = {'before.{0}.': 'after.{(lambda x: x.replace("a", "aa"))("[0]")}.'}
            >>> Converter.convert_keys(state_dict, convert_dict)
            OrderedDict([('after.aa.1.weight', 1), ('after.b.2.weight', 2)])

        """
        from .nlp_utils import PrefixTree

        def parse(s):
            """split the string with wildcards, and retrun their indexes
            Examples:
                # key is wildcard's values, idx is start index of wildcard in string
                >>> parse('a.{0}.a')
                (('a', '.', '{0}', '.', 'a'), {'key': ['_0'], 'idx': [2], 'ops': ['']})

                >>> parse('a.{[0]+1}.a')
                (('a', '.', '{0}', '.', 'a'), {'key': ['_0'], 'idx': [2], 'ops': ['[0]+1']})
            """
            match = re.findall('\{.+?\}', s)
            end, tmp, spans, key, ops = 0, s, [], [], []
            for m in match:
                op = ''
                start = tmp.index(m) + end
                end = start + len(m)
                spans.append((start, end))
                k = m[1:-1]

                r = re.findall('\[(.+?)\]', k)
                if r:
                    op = k
                    for i, rr in enumerate(r):
                        if rr.isdigit():
                            op = op.replace(f'[{rr}]', f'[_{rr}]')
                            r[i] = '_' + rr
                    k = '#'.join(r)

                if k.isdigit():
                    # digit str would raise error by f'(?P<{_k}>.+?)'
                    k = '_' + k

                key.append(k)
                ops.append(op)
                tmp = s[end:]

            r = []
            end = 0
            idx = []
            for i, span in enumerate(spans):
                start, end1 = span
                tmp = list(s[end:start])
                r += tmp + [match[i]]
                idx.append(len(r) - 1)
                end = end1

            r += list(s[end: len(s)])
            return tuple(r), {'key': key, 'idx': idx, 'ops': ops}

        split_convert_dict = {}
        a_values, b_values = {}, {}
        for a, b in convert_dict.items():
            a_key, a_value = parse(a)
            b_key, b_value = parse(b)
            split_convert_dict[a_key] = b_key
            a_values[a_key] = a_value
            b_values[b_key] = b_value

        tree = PrefixTree(list(split_convert_dict.keys()), list(split_convert_dict.keys()), unique_value=True)
        d = OrderedDict()

        for k, v in state_dict.items():
            # find string before convert
            a = tree.get(k, return_last=True)
            if a in split_convert_dict:
                # make string after convert
                b = split_convert_dict[a]
                a_value = a_values[a]
                b_value = b_values[b]

                p, pa = '', ''
                for i, s in enumerate(a):
                    if i in a_value['idx']:
                        ii = a_value['idx'].index(i)
                        _k = a_value["key"][ii]
                        p += f'(?P<{_k}>.+?)'
                        pa += '{' + _k + '}'
                    else:
                        p += '\\' + s if s == '.' else s
                        pa += s

                if a_value['idx']:
                    ra = re.search(p, k)
                    r = ra.groupdict()
                    pa = pa.format(**r)

                    b = list(b)
                    for key, idx, op in zip(b_value['key'], b_value['idx'], b_value['ops']):
                        if '#' in key:
                            keys = key.split('#')
                            for key in keys:
                                _v = r[key]
                                op = op.replace(f'[{key}]', f"{_v}")
                            _v = str(eval(op))
                        else:
                            _v = r[key]
                            if op:
                                _v = str(eval(op.replace(f'[{key}]', f"{_v}")))

                        b[idx] = _v

                # replace before string to after string
                pb = ''.join(b)
                k = k.replace(pa, pb)
            d[k] = v

        return d

    @staticmethod
    def conv_weight_from_tf_to_torch(weight):
        """(h, w, c, n) -> (n, c, h, w)"""
        return weight.permute(3, 2, 0, 1)

    @staticmethod
    def dw_conv_weight_from_tf_to_torch(weight):
        """(h, w, n, c) -> (n, c, h, w)"""
        return weight.permute(2, 3, 0, 1)

    @staticmethod
    def linear_weight_from_tf_to_torch(weight):
        """(n, h, w) -> (w, h)"""
        if len(weight.size()) == 3:
            weight = weight.squeeze(0)
        if len(weight.size()) == 2:
            weight = weight.permute(1, 0)
        return weight

    @classmethod
    def make_convert_tf_funcs(cls):
        return {
            'c': cls.conv_weight_from_tf_to_torch,
            'gc': cls.dw_conv_weight_from_tf_to_torch,
            'l': cls.linear_weight_from_tf_to_torch,
        }

    torch_tensor_types = {
        'w': 'weight',
        'b': 'bias',
        'g': 'gamma',
        'bt': 'beta',
        'nm': 'running_mean',  # norm mean
        'nv': 'running_var',  # norm var
    }

    @classmethod
    def tensors_from_tf_to_torch(cls, state_dict, key_types=None, value_types=None) -> OrderedDict:
        """

        Args:
            state_dict (OrderedDict | dict):
            key_types (list): see `torch_tensor_types`
            value_types (list): only work when key_type is 'w',see `make_convert_tf_funcs`

        Examples
            >>> Converter.tensors_from_tf_to_torch({'ln_1/g:0': 0}, key_types=['w'])
            OrderedDict([('ln_1.weight', 0)])

            >>> Converter.tensors_from_tf_to_torch({'ln_1/g:0': 0}, key_types=['g'])
            OrderedDict([('ln_1.gamma', 0)])

            >>> Converter.tensors_from_tf_to_torch({'ln_1/g:0': torch.zeros((2, 1))}, key_types=['w'], value_types=['l'])
            OrderedDict([('ln_1.weight', tensor([[0., 0.]]))])
            # shape: (2, 1) -> (1, 2)

        """

        key_types = key_types or [''] * len(state_dict)
        value_types = value_types or [''] * len(state_dict)

        assert len(key_types) == len(state_dict)
        assert len(value_types) == len(state_dict)

        d = OrderedDict()
        convert_tf_funcs = cls.make_convert_tf_funcs()

        for i, (k, v) in enumerate(state_dict.items()):
            # convert key
            tmp = re.split(r'[/\.]', k)
            if len(tmp) > 1:
                suffix = tmp[-1]
                new_suffix = cls.torch_tensor_types.get(key_types[i], suffix)
                k = '.'.join(tmp[:-1]) + '.' + new_suffix

            # convert value
            if key_types[i] == 'w' and value_types[i] in convert_tf_funcs:
                v = convert_tf_funcs[value_types[i]](v)
            d[k] = v

        return d

    @classmethod
    def make_convert_paddle_funcs(cls):
        return {
            # 'c': cls.conv_weight_from_paddle_to_torch,
            'l': cls.linear_weight_from_paddle_to_torch,
        }

    # @staticmethod
    # def conv_weight_from_paddle_to_torch(weight):
    #     """(n, c, w, h) -> (n, c, h, w)"""
    #     return weight.permute(0, 1, 3, 2)

    @staticmethod
    def linear_weight_from_paddle_to_torch(weight):
        return weight.T

    @classmethod
    def tensors_from_paddle_to_torch(cls, state_dict, key_types=None, value_types=None) -> OrderedDict:
        key_types = key_types or [''] * len(state_dict)
        value_types = value_types or [''] * len(state_dict)

        assert len(key_types) == len(state_dict)
        assert len(value_types) == len(state_dict)

        d = OrderedDict()
        convert_tf_funcs = cls.make_convert_paddle_funcs()

        for i, (k, v) in enumerate(state_dict.items()):
            # convert key
            if key_types[i] == 'nm':
                k = k.replace('._mean', '.running_mean')
            elif key_types[i] == 'nv':
                k = k.replace('._variance', '.running_var')

            # convert value
            if key_types[i] == 'w' and value_types[i] in convert_tf_funcs:
                v = convert_tf_funcs[value_types[i]](v)
            d[k] = v

        return d


def make_optimizer_cls(name: str):
    if name in {"Lion"}:
        import lion_pytorch  # pip install lion_pytorch
        return getattr(lion_pytorch, name)

    elif name in {
        'AdamW8bit', 'SGDNesterov8bit', 'Lion8bit', 'PagedAdamW8bit', 'PagedLion8bit',
        'PagedAdamW', 'PagedAdamW32bit'
    }:
        import bitsandbytes as bnb
        return getattr(bnb.optim, name)

    elif name in {'DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptAdan', 'DAdaptLion', 'DAdaptSGD'}:
        import dadaptation
        return getattr(dadaptation, name)

    elif name in {'DAdaptAdamPreprint', 'DAdaptAdanIP'}:
        from dadaptation import experimental
        return getattr(experimental, name)

    elif name in {'Prodigy'}:
        import prodigyopt
        return getattr(prodigyopt, name)

    elif name in {'SGD', 'AdamW', }:
        return getattr(torch.optim, name)

    elif name in {'Adafactor'}:
        from transformers import optimization
        return getattr(optimization, name)

    else:
        raise ValueError(f'Do not support optimizer with name of {name}')


class SchedulerMaker:
    @classmethod
    def lr_scheduler(cls, name, optimizer, last_epoch=-1, **kwargs):
        from torch.optim.lr_scheduler import LambdaLR

        lr_lambda = getattr(cls, name)(**kwargs)
        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    @staticmethod
    def constant_lr_lambda(**kwargs):
        """
        Create a schedule with a constant learning rate, using the learning rate set in optimizer.
        """
        return lambda _: 1

    @staticmethod
    def constant_with_warmup_lr_lambda(num_warmup_steps=100, **kwargs):
        """
        Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
        increases linearly between 0 and the initial lr set in the optimizer.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        return lr_lambda

    @staticmethod
    def piecewise_constant_lr_lambda(step_rules=None, **kwargs):
        """
        Create a schedule with a constant learning rate, using the learning rate set in optimizer.

        Args:
            step_rules (`str`):
                The rules for the learning rate. ex: rule_steps="1:10,0.1:20,0.01:30,0.005" it means that the learning rate
                if multiple 1 for the first 10 steps, mutiple 0.1 for the next 20 steps, multiple 0.01 for the next 30
                steps and multiple 0.005 for the other steps.

        """
        rules_dict = {}
        rule_list = step_rules.split(",")
        for rule_str in rule_list[:-1]:
            value_str, steps_str = rule_str.split(":")
            steps = int(steps_str)
            value = float(value_str)
            rules_dict[steps] = value
        last_lr_multiple = float(rule_list[-1])

        def create_rules_function(rules_dict, last_lr_multiple):
            def rule_func(steps: int) -> float:
                sorted_steps = sorted(rules_dict.keys())
                for i, sorted_step in enumerate(sorted_steps):
                    if steps < sorted_step:
                        return rules_dict[sorted_steps[i]]
                return last_lr_multiple

            return rule_func

        rules_func = create_rules_function(rules_dict, last_lr_multiple)

        return rules_func

    @staticmethod
    def linear_with_warmup_lr_lambda(num_warmup_steps: int = None, num_training_steps: int = None, **kwargs):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return lr_lambda

    @staticmethod
    def cosine_lr_lambda(max_epoch=-1, lrf=0.01, **kwargs):
        return lambda x: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1

    @staticmethod
    def cosine_with_warmup_lr_lambda(num_warmup_steps: int = None, num_training_steps: int = None, num_cycles: float = 0.5, **kwargs):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.

        Args:
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`float`, *optional*, defaults to 0.5):
                The number of periods of the cosine function in a schedule (the default is to just decrease from the max
                value to 0 following a half-cosine).

        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return lr_lambda

    @staticmethod
    def cosine_with_hard_restarts_with_warmup_lr_lambda(num_warmup_steps: int = None, num_training_steps: int = None, num_cycles: int = 1, **kwargs):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
        linearly between 0 and the initial lr set in the optimizer.

        Args:
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`int`, *optional*, defaults to 1):
                The number of hard restarts to use.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            if progress >= 1.0:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

        return lr_lambda

    @staticmethod
    def polynomial_decay_with_warmup_lr_lambda(
            num_warmup_steps: int = None,
            num_training_steps: int = None,
            lr_init: float = None,
            lr_end: float = 1e-7,
            power: float = 1.0,
            **kwargs
    ):
        """
        Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
        optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
        initial lr set in the optimizer.

        Args:
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            lr_init:
                can get by `optimizer.defaults["lr"]`
            lr_end (`float`, *optional*, defaults to 1e-7):
                The end LR.
            power (`float`, *optional*, defaults to 1.0):
                Power factor.

        Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
        implementation at
        https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
        """
        if not (lr_init > lr_end):
            raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step > num_training_steps:
                return lr_end / lr_init  # as LambdaLR multiplies by lr_init
            else:
                lr_range = lr_init - lr_end
                decay_steps = num_training_steps - num_warmup_steps
                pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
                decay = lr_range * pct_remaining ** power + lr_end
                return decay / lr_init  # as LambdaLR multiplies by lr_init

        return lr_lambda

    @staticmethod
    def adafactor_scheduler(name, optimizer, **kwargs):
        import transformers

        assert isinstance(optimizer, transformers.optimization.Adafactor), "adafactor scheduler must be used with Adafactor optimizer"

        initial_lr = float(name.split(":")[1])
        return transformers.optimization.AdafactorSchedule(optimizer, initial_lr)
