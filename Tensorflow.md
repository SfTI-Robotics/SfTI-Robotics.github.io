# Tensorflow

##Tensor
Tensorflow uses tensor (data structure) to represent all data. Tensors are the only inputs passed into a tensorflow computational graph. It can be thought of as a n-dimensional array or list. A tensor has a rank, a shape and a static type, so a tensor can be represented as a multidimensional array of numbers.

-Rank: number of dimensions in a tensor. 
-Scalar: rank 0
-Vector: rank 1 
-matrix : rank 2
Shape: dimension lengths in a tensor 
2D matrix shape returns: [rows, columns]
3D tensor shape returns [rows, columns, width]
Data type: 
Tf.float32
Tf.int64
etc...
Variables
in-memory buffers containing tensors. They can hold and update parameters when training models. Variables maintain state across executions of the graph.
Fetches
returns the output of operations by executing the graph with a sess.run() call. 
Feeds  
this mechanism patches tensors directly into any operation in the graph. It temporarily replaces the output of an operation with a tensor value. When the run() function is called, the feed supplies input data as an argument. Common feeds are the tf.placeholder() function
tf.__ functions
