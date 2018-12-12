# Tensorflow

## What is TensorFlow?
TensorFlow is a open source library developed by Google to run large numerical computations. We use it in machine learning to create and operation neural networks. Tensorflow runs on both CPU and GPU. As neural network is not the primary focus of this project, we will mainly be using Keras, a high level API that runs on top of TensorFlow which helps simplify coding with TensorFlow. 

## Why use TensorFlow?
We don't want to worry about the nitty-gritties of how to create a neural network from scratch. TensorFlow also offers a wide range of commands and tools for manipulating the neural network.

## Tensor
Tensor is what TensorFlow processes. You can think of it as a set of values that is used to represent something mathematical. It can be thought of as a n-dimensional array. Take a 2D vector as an example. We can represent this vector by recording it's x and y axis values. This set of values, ie [3, 4] is a tensor of rank 1. A tensor has a rank, a shape and a static type, so a tensor can be represented as a multidimensional array of numbers.

- Rank: number of dimensions in a tensor. 
  - Scalar: rank 0
  - Vector: rank 1 
  - matrix : rank 2
  
- Shape: dimension lengths in a tensor 
  - 2D matrix shape returns: [rows, columns]
  - 3D tensor shape returns [depth, rows, columns]
  
- Data type: 
  - tf.float32
  - tf.int64
  - etc...

[Helpful video (12 mins)](https://www.youtube.com/watch?v=f5liqUk0ZTw)

# Low Level API
## Dataflow Graphs
TensorFlow uses dataflow graphs to represent computation. It is very useful to visualize relation between operations.
![](https://cdn-images-1.medium.com/max/1600/1*zeXlzGhBoCl8clrpwtVbRQ.png)

## Using TensorFlow
There are 2 steps when programming with TensorFlow. We need to first build our computational graph, then run it. A computational graph is a series of TensorFlow operations arranged into a graph. 
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

reshape(): returns the tensor in the shape specified
```
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]
 # tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]
 # pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
 # -1 can also be used to infer the shape
 # -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
                         
# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
```
