import tensorflow as tf

from handlers.backend_handler import BackendHandler
from handlers.handler import onnx_op
from handlers.handler import tf_func
from .math_mixin import BasicMathMixin


@onnx_op("Sinh")
@tf_func(tf.sinh)
class Sinh(BasicMathMixin, BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
