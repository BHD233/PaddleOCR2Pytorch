import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch

from pytorchocr.modeling.architectures.base_model import BaseModel

class BaseOCRV20:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)


    def load_paddle_weights(self, weights_path):
        raise NotImplementedError('implemented in converter.')
        print('paddle weights loading...')
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)

        for k,v in self.net.state_dict().items():
            name = k

            if name.endswith('num_batches_tracked'):
                continue

            if name.endswith('running_mean'):
                ppname = name.replace('running_mean', '_mean')
            elif name.endswith('running_var'):
                ppname = name.replace('running_var', '_variance')
            elif name.endswith('bias') or name.endswith('weight'):
                ppname = name
            elif 'lstm' in name:
                ppname = name

            else:
                print('Redundance:')
                print(name)
                raise ValueError
            try:
                if ppname.endswith('fc.weight'):
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
                else:
                    self.net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
                raise e

        print('model is loaded: {}'.format(weights_path))

    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = torch.load(weights_path)
        return weights

    def get_out_channels(self, weights):
        out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        print('weighs is loaded.')

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        print('model is loaded: {}'.format(weights_path))


    def save_pytorch_weights(self, weights_path):
        try:
            torch.save(self.net.state_dict(), weights_path, _use_new_zipfile_serialization=False)
        except:
            torch.save(self.net.state_dict(), weights_path) # _use_new_zipfile_serialization=False for torch>=1.6.0
        print('model is saved: {}'.format(weights_path))

    def save_onnx(self, path, type = 'det'):
        print('saving model', type)
        if type == 'det':
            # ------------------------------ text detection
            # Input to the model
            x = torch.randn(1, 3, 960, 1280, requires_grad=True)

            # Export the model
            torch.onnx.export(self.net,             # model being run
                            x,                         # model input (or a tuple for multiple inputs)
                            path,  # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=12,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size', 
                                                    2 : 'height_size', 
                                                    3 : 'width_size'},    # variable lenght axes
                                            'output' : {0 : 'batch_size', 
                                                        2 : 'height_size', 
                                                        3 : 'width_size'}})

        if type == 'cls':
            # ------------------------------- class detection
            # Input to the model
            x = torch.randn(1, 3, 48, 192, requires_grad=True)

            # Export the model
            torch.onnx.export(self.net,             # model being run
                            x,                         # model input (or a tuple for multiple inputs)
                            path,  # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=12,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                            'output' : {0 : 'batch_size'}})
        if type == 'rec':
            # ------------------------------ recog
            # Input to the model
            x = torch.randn(1, 3, 32, 320, requires_grad=True)

            # Export the model
            torch.onnx.export(self.net.eval(),             # model being run
                            x,                         # model input (or a tuple for multiple inputs)
                            path,  # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=12,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            dynamic_axes={'input' : {0 : 'batch_size', 
                                                    3 : 'width_size'},    # variable lenght axes
                                            'output' : {0 : 'batch_size', 
                                                        1 : 'width_size'}})

    def print_pytorch_state_dict(self):
        print('pytorch:')
        for k,v in self.net.state_dict().items():
            print('{}----{}'.format(k,type(v)))

    def read_paddle_weights(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        return para_state_dict, opti_state_dict

    def print_paddle_state_dict(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        print('paddle"')
        for k,v in para_state_dict.items():
            print('{}----{}'.format(k,type(v)))


    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer
