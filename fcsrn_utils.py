import numpy as np
import tensorflow as tf

def label_to_array(label, char_vector):
    try:
        return [char_vector.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex

def sparse_tuple_from(sequences, dtype=np.int32,tf_type=True):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
        Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """


    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray(
        [len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64
    )
    if tf_type:
        return tf.SparseTensor(indices,values,shape)
    else:
        return indices, values, shape  


if __name__ == '__main__':
    char_vector = 'abcdefg'
    test = label_to_array(['a','a','b','c'],char_vector)
    test_tuple = [[0,0,1,2],[2,2,1,0]]
    test_batch = np.reshape(np.array(test_tuple), (-1))
    print(test_batch)
    batch_dt = sparse_tuple_from(test_tuple)
    print(batch_dt)