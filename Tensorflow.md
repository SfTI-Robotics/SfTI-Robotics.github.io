# Tensorflow

## What is TensorFlow?
TensorFlow is a open source library developed by Google to run large numerical computations. We use it in machine learning to create and operation neural networks. Tensorflow can perform parallel calculations with either CPU or GPU, allowing for significant reduction in computation time. As neural network is not the primary focus of this project, you may want to also look into Keras, a high level API that runs on top of TensorFlow which helps simplify coding with TensorFlow. 

## Why use TensorFlow?
We don't want to worry about the nitty-gritties of how to create a neural network from scratch. TensorFlow also offers a wide range of commands and tools for manipulating the neural network.

## Importing TensorFlow
You can download tensorflow from their offical website. After installing either the GPU or CPU version for python 2.7/3, import it with:
```
import tensorflow as tf
```

# Tensor
Tensor is what TensorFlow processes. It is a generalization of vectors and matrices to any dimension. It can be thought of as a n-dimensional array. Take a 2D vector as an example. We can represent this vector by recording it's x and y axis values. This set of values, ie [3, 4] is a tensor of rank 1. We use tensors because they offer a more natural representations of data.

- Rank: number of dimensions in a tensor. 
  - Scalar: rank 0. Has magnitude but no direction.
  - Vector: rank 1. Has both magnitude and direction. May represent a vector in 2,3 or n dimensions
  - matrix : rank 2. Can be thought of as the possible combination of two vectors. 
  
- Shape: dimension lengths in a tensor 
  - 2D matrix shape returns: [rows, columns]
  - 3D tensor shape returns [depth, rows, columns]

Tensors have both an inferred (static) shape and a true (dynamic) shape. 
Return shape of tensor
```
tensor.get_shape() # static shape
tensor.shape() # dynamic shape
```
  
- Data type: 
  - tf.float32  # This is the most common data type when using tensorflow.
  - tf.int64
  - etc...

[Helpful Video that Explain Tensors (12 mins)](https://www.youtube.com/watch?v=f5liqUk0ZTw)

The main objective of a TensorFlow programme is to manipulate and pass around tensors through mathematical operations. Tensors are represented with tf.Tensor objects. It represents a partially defined computation that will eventually produce a value. TensorFlow programs work by first building a graph of tf.Tensor objects, detailing how each tensor is computed based on the other available tensors and then by running parts of this graph to achieve the desired results.

![](http://adventuresinmachinelearning.com/wp-content/uploads/2017/03/TensorFlow-data-flow-graph.gif)

# Core
TensorFlow programme consist of two discrete sections:
  1. Building the computational graph
  2. Running the computational graph

## Building the computation graph (tf.Graph)
A series of TensorFlow operations are arranged into a graph. The graph is composed of both operations (nodes) and tensors (edges). Reminder that the tensor objects do not hold values, they are just handles to elements in the computation graph.

## Running the computation graph (tf.Session)
We create a Session() object to evaluate the graph. Each session.run is a separate execution, so take note on what you run. 

Run a session (using built model)
```
with tf.Session() as sess:
  sess.run()
  
or

sess = tf.Session()
sess.run()
```

## Feeding 
Placeholders are how we parameterise the computation graph to accept external inputs
```
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

# Layers
A trainable model modify the values in the graph to get new outputs with the same input. tf.layer add tranable parameters to a graph. 
Layer package together both the variables and the operations that act on them. The densely-connected layer performs a weihte sum across all inputs for each output and applies an activation function. The weight and biases are managed by te layer object.

Process to create a dense layer are as below:
  1. Create layers
```
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)
```
  2. Initialize layers
```
init = tf.global_variables_initializer()
sess.run(init)
```
  3. Execute layers
```
sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]})
```

# Syntax
## Creating a Tensor
```
tf.constant() # value does not change
```
Create a Tensor placeholder
```
tf.placeholder()
```
Create a Tensor with all zeros
```
tf.zeros()
```
Create a Tensor with all ones
```
tf.ones()
```
Create a Tensor with random values from a normal distribution
```
tf.random_normal()
```
Create a Tensor with random values with a uniform distribution
```
tf.random_uniform()
```

## Variables
A `tf.Variable` represents a tensor whose value can be changed by running operations (ops) on it. They can hold and update parameters when training models. Variables maintain state across executions of the graph. Unlike tf.Tensor objects, a tf.Variable exists outside the context of a single session.run call.

Creating a variable
```
# constructor which will create a new variable every time it is called (and potentially add a 
# suffix to the variable name if a variable with such name already exists).
tf.Variable() 

# create a new variable with such name or retrieve the one that was created before
tf.get_variable() 
```

## Namespace
A way to organise names for variables and operators in a hierachical manner. Variables can be accessed in different parts of the code without passing references to the variable, through the mechanism of variable sharing. Scope for the variable are added as a prefix to the operation or variable name.

### Variable scope
`tf.variable_scope` creates namespace for both variables and operators in the default graph.
```
with tf.variable_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # my_scope/var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0
```

### Name scope
Creates namespace for operators in the default graph.
`tf.name_scope` ignores variables created by the `tf.get_variable` operation
```
with tf.name_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0
```
```
with tf.variable_scope('q_target'):
     # update q-target
     q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    
```

### tf.variable_scope vs tf.name_scope
`tf.variable_scope` is more comprehensive since it can access all variables in the graph. 
`tf.name_scope` can only access variables created by `tf.Variable`.


## What is a Kernel

In machine learning, a “kernel” is usually used to refer to the kernel trick, a method of using a linear classifier to solve a non-linear problem. 



## Matrix Manipulation

Multiply matrix
```
tf.matmul(matrix1, matrix2)
```

## tf.__ functions

shape(): Returns the the dimensions of a single tensor
shape_n(): Returns the dimensions of tensors
```
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.shape(t)  # [2, 2, 3] 
# shape() need to fully explain example
```

stack(): packs the list of tensors into one tensor and also increases the rank by one. It is stacked along the axis dimension. 
```
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
```

assign(): This operation outputs a Tensor that holds the new value of 'ref' after the value has been assigned. This makes it easier to chain operations that need to use the reset value.
```
# t is assigned the value of the current e variable
tf.assign(t, e)
```

reshape(): returns the tensor in the shape specified
```
# tensor, t = [1, 2, 3, 4, 5, 6, 7, 8, 9],  shape = [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]

# tensor 't' = [[[1, 1], [2, 2]],
#              [[3, 3], [4, 4]]]
# shape = [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]

# tensor 't' = [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
```

stop_gradient(): this operation provides a way to not compute gradient with respect to some variables during back-propagation. It acts as the identity function in the forward direction, but stops the accumulated gradient from flowing through that operator in the backward direction. It does not prevent backpropagation altogether, but instead prevents an individual tensor from contributing to the gradients that are computed for an expression. 

Example: we have three variables, `w1, w2, w3` and input x. 
The loss is `square((x1.dot(w1) - x.dot(w2 * w3)))`. We want to minimize this loss wrt to w1 but want to keep w2 and w3 fixed. To achieve this we can just put `tf.stop_gradient(tf.matmul(x, w2*w3))`.

```
w1 = tf.get_variable("w1", shape=[5, 1], initializer=tf.truncated_normal_initializer())
w2 = tf.get_variable("w2", shape=[5, 1], initializer=tf.truncated_normal_initializer())
w3 = tf.get_variable("w3", shape=[5, 1], initializer=tf.truncated_normal_initializer())
x = tf.placeholder(tf.float32, shape=[None, 5], name="x")


a1 = tf.matmul(x, w1)
a2 = tf.matmul(x, w2*w3)
a2 = tf.stop_gradient(a2)

loss = tf.reduce_mean(tf.square(a1 - a2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
gradients = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(gradients)
```

gather_nd(): 

reduce_mean(

hasattr()

# tf.layers.__


dense(): 

`tf.layers.dense` 
```
# layer 1: input = s
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            # layer 2: input = e1
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')
```
