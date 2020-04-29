import tensorflow as tf

from handlers.backend_handler import BackendHandler
from handlers.handler import onnx_op
from handlers.handler import tf_func


@onnx_op("Det")
@tf_func(tf.linalg.det)
class Det(BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    return [
        cls.make_tensor_from_onnx_node(node, **kwargs)
    ]
