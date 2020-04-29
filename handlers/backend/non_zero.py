import tensorflow as tf

from handlers.backend_handler import BackendHandler
from handlers.handler import onnx_op


@onnx_op("NonZero")
class NonZero(BackendHandler):

  @classmethod
  def version_9(cls, node, **kwargs):
    input_tensor = kwargs["tensor_dict"][node.inputs[0]]
    condition = tf.not_equal(input_tensor, tf.zeros_like(input_tensor))
    nonzero_indices = tf.where(condition)
    return [tf.transpose(nonzero_indices)]
