import tensorflow as tf

from handlers.backend_handler import BackendHandler
from handlers.handler import onnx_op
from handlers.handler import tf_func
from .gather_and_scatter_mixin import GatherAndScatterMixin


@onnx_op("ScatterND")
@tf_func(tf.tensor_scatter_nd_update)
class ScatterND(GatherAndScatterMixin, BackendHandler):

  @classmethod
  def version_11(cls, node, **kwargs):
    data = kwargs["tensor_dict"][node.inputs[0]]
    indices = kwargs["tensor_dict"][node.inputs[1]]
    updates = kwargs["tensor_dict"][node.inputs[2]]

    result = cls.chk_idx_out_of_bounds(data, indices)
    msg = 'ScatterND indices are out of bounds, please double check the indices and retry.'
    with tf.control_dependencies(
        [tf.compat.v1.assert_equal(result, True, message=msg)]):
      indices = cls.process_neg_idx(data, indices)
      return [
          cls.make_tensor_from_onnx_node(node,
                                         inputs=[data, indices, updates],
                                         **kwargs)
      ]
