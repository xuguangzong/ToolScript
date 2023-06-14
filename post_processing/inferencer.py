import os
import onnx
import pycuda.driver as cuda
import tensorrt as trt
from onnx import ModelProto
import onnxruntime

from utils.data_class import ModelConfig, HostDeviceMem


class Inferencer:
    """
    推理器
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.already_initialized = False  # 是否已经初始化，第一次推理的时候才会进行初始化，初始化调用init方法

    def init(self):
        """
        初始化，申请资源
        :return:
        """
        raise NotImplementedError

    def uninit(self):
        """
        反初始化
        :return:
        """
        raise NotImplementedError

    def infer(self):
        """
        推理代码
        :return:
        """
        raise NotImplementedError

    def prepare(self):
        """
        每一帧推理之前调用，确认已经完成了初始化，如果没有初始化就进行初始化
        :return:
        """
        try:
            if not self.already_initialized:
                self.init()
        except AttributeError:
            raise Exception(f"没有找到模型进行初始化：{self.__class__.__name__}")


class TensorRTInferencer(Inferencer):

    def __init__(self, model_config: ModelConfig):
        """
        TRT推理
        """
        super().__init__(model_config=model_config)
        self.model_config = model_config

        self.trt_logger = None
        self.engine = None
        self.context = None
        self.cuda_ctx = None
        self.inputs, self.outputs, self.bindings, self.stream = None, None, None, None

    # noinspection PyArgumentList
    def init(self):
        """
        初始化
        :return:
        """
        cuda.init()
        self.cuda_ctx = cuda.Device(self.model_config.device).make_context()
        # noinspection PyUnresolvedReferences
        self.trt_logger = trt.Logger(trt.Logger.ERROR)
        self.engine = self.get_engine(str(self.model_config.onnx_file), str(self.model_config.engine_file))
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.cuda_ctx.pop()
        self.already_initialized = True

    def uninit(self):
        """
        反初始化，释放资源
        :return:
        """
        if not self.already_initialized:
            return
        del self.context
        del self.engine
        del self.cuda_ctx
        del self.inputs
        del self.outputs
        del self.bindings
        del self.stream
        self.already_initialized = False

    def infer(self):
        raise NotImplementedError

    def __del__(self):
        self.uninit()

    def get_engine(self, onnx_file: str, engine_file: str):
        """
        获取engine,优先找trt文件，找不到就在线onnx转trt
        :param onnx_file:
        :param engine_file:
        :return:
        """
        if os.path.exists(engine_file):
            # print(f"读取trt engine {engine_file}")
            with open(engine_file, "rb") as f:
                engine_data = f.read()
            # noinspection PyUnresolvedReferences
            return trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_data)
        else:
            print(f"trt engine {engine_file}不存在，开始在线转换 {onnx_file}")
            # noinspection PyUnresolvedReferences
            explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            # noinspection PyUnresolvedReferences
            with trt.Builder(self.trt_logger) as builder, \
                    builder.create_network(explicit_batch) as network, \
                    trt.OnnxParser(network, self.trt_logger) as parser:

                builder.max_workspace_size = 1 << 30
                builder.max_batch_size = 1
                if builder.platform_has_fast_fp16:
                    builder.fp16_mode = True
                with open(onnx_file, 'rb') as model_file:
                    model_data = model_file.read()
                    if not model_data:
                        raise Exception(f"onnx文件为空：{onnx_file}")
                    if not parser.parse(model_data):
                        print('Failed to parse the ONNX file')
                        for err in range(parser.num_errors):
                            print(parser.get_error(err))
                        return None
                    model = ModelProto()
                    model.ParseFromString(model_data)
                    onnx_input = model.graph.input
                input_shapes = []
                for i in range(len(onnx_input)):
                    input_shape = []
                    input_dim = len(onnx_input[i].type.tensor_type.shape.dim)
                    for j in range(input_dim):
                        dim = onnx_input[i].type.tensor_type.shape.dim[j].dim_value
                        input_shape.append(dim)
                    if len(input_shape):
                        input_shapes.append(input_shape)
                # 这里是可以直接从onnx里拿模型的shape然后直接转换的，
                # 进球检测的模型的输入只有一个，但是这里拿到100多个，怀疑是mm的问题
                # 正常的模型一般都是一个输入，slow fast有俩输入，这里暂时先hack一下，如果输入已经超过50了，就拿第一个
                input_shapes = input_shapes[:1] if len(input_shapes) > 50 else input_shapes
                for index, input_size in enumerate(input_shapes):
                    network.get_input(index).shape = input_shapes[index]
                engine = builder.build_cuda_engine(network)
                if engine is None:
                    print('Failed to build engine')
                    return None
                with open(engine_file, 'wb') as engine_file:
                    engine_file.write(engine.serialize())
                return engine

    # noinspection PyArgumentList
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            # noinspection SpellCheckingInspection
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self):
        self.cuda_ctx.push()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        result = [out.host for out in self.outputs]
        self.cuda_ctx.pop()
        return result


class OnnxInferencer(Inferencer):
    """
    onnx推理
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config=model_config)
        self.onnx_session = None

    def init(self):
        # 两种写法都可以
        # self.model = onnx.load_model(self.model_config.onnx_file.as_posix())
        # self.onnx_session = onnxruntime.InferenceSession(self.model.ParseFromString(),
        #                                                  providers=["CUDAExecutionProvider"])
        # self.input_name = [i.name for i in self.sess.get_inputs()[:]]
        # self.output_names = [i.name for i in self.sess.get_outputs()[:]]
        self.onnx_session = onnxruntime.InferenceSession(str(self.model_config.onnx_file),
                                                         providers=["CUDAExecutionProvider"])
        self.already_initialized = True

    def uninit(self):
        raise NotImplementedError

    def infer(self):
        raise NotImplementedError
