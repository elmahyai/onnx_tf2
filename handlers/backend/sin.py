import tensorflow as tf

from handlers.backend_handler import BackendHandler
from handlers.handler import onnx_op
from handlers.handler import tf_func
from .math_mixin import BasicMathMixin


@onnx_op("Sin")
@tf_func(tf.sin)
class Sin(BasicMathMixin, BackendHandler):

  @classmethod
  def version_7(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]