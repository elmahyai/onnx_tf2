from handlers.backend_handler import BackendHandler
from handlers.handler import onnx_op
from handlers.handler import partial_support
from handlers.handler import ps_description
from .conv_mixin import ConvMixin


@onnx_op("ConvTranspose")
@partial_support(True)
@ps_description("ConvTranspose with dilations != 1, or " +
                "transposed convolution for 4D or higher " +
                "are not supported in Tensorflow.")
class ConvTranspose(ConvMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.conv(node, kwargs["tensor_dict"], transpose=True)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls.conv(node, kwargs["tensor_dict"], transpose=True)
