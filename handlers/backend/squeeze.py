import tensorflow as tf

from handlers.backend_handler import BackendHandler
from handlers.handler import onnx_op
from handlers.handler import tf_func


@onnx_op("Squeeze")
@tf_func(tf.squeeze)
class Squeeze(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"rename": {"axes": "axis"}}

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_11(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
