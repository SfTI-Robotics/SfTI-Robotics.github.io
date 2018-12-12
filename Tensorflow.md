# Tensorflow

## Tensor
Tensorflow uses tensor (data structure) to represent all data. Tensors are the only inputs passed into a tensorflow computational graph. It can be thought of as a n-dimensional array or list. A tensor has a rank, a shape and a static type, so a tensor can be represented as a multidimensional array of numbers.

- Rank: number of dimensions in a tensor. 
  - Scalar: rank 0
  - Vector: rank 1 
  - matrix : rank 2
- Shape: dimension lengths in a tensor 
  - 2D matrix shape returns: [rows, columns]
  - 3D tensor shape returns [rows, columns, width]
- Data type: 
  - Tf.float32
  - Tf.int64
  - etc...

## Variables
in-memory buffers containing tensors. They can hold and update parameters when training models. Variables maintain state across executions of the graph.

## Fetches
returns the output of operations by executing the graph with a sess.run() call. 

## Feeds  
this mechanism patches tensors directly into any operation in the graph. It temporarily replaces the output of an operation with a tensor value. When the run() function is called, the feed supplies input data as an argument. Common feeds are the tf.placeholder() function

##tf.__ functions

shape(): Returns the the dimensions of a single tensor
shape_n(): Returns the dimensions of tensors
```
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.shape(t)  # [2, 2, 3] 
```

stack(): packs the list of tensors into one tensor and also increases the rank by one. It is stacked along the axis dimension. 
```
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
```

variable_scope(): created variables can be contained within a specified scope in the computational program. It allows for variable sharing, which avoids passing references to the variable itself. 
```
with tf.variable_scope('q_target'):
     # update q-target
     q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    
```
Assign(): This operation outputs a Tensor that holds the new value of 'ref' after the value has been assigned. This makes it easier to chain operations that need to use the reset value.
