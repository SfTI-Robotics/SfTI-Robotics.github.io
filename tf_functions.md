# TensorFlow Functions
## tf.__
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
Create a Tensor with random values from a truncated normal distribution.
```
tf.truncated_normal
```
Multiple matrices
```
tf.matmul(matrix1, matrix2)
```
shape(): Returns the the dimensions of a single tensor
shape_n(): Returns the dimensions of tensors
```
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.shape(t)  # [2, 2, 3] 
tf.shape_n([t]) # [array([2, 2, 3], dtype=int32)]
```

stack(): packs the list of tensors into one tensor and also increases the rank by one. It is stacked along the axis dimension. 
```
x = tf.constant([1, 4]) # These arrays have a dimension of 1
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

stop_gradient(): this operation provides a way to not compute gradient with respect to some variables during back-propagation. It acts as the identity function in the forward direction, but stops the accumulated gradient from flowing through that operator in the backward direction. It does not prevent backpropagation altogether, but instead prevents an individual tensor from contributing to the gradients that are computed for an expression. ie. it restrict the flow of gradients through certain parts of the network

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

gather_nd(): Gather slices from params into a Tensor with shape specified by indices.
```
 indices = [[1], [0]]
 params = [['a', 'b'], ['c', 'd']]
 
 output = tf.gather_nd(params,indices) # output: [['c', 'd'], ['a', 'b']]
```
reduce_mean(): Computes the mean of elements across dimensions of a tensor
```
x = tf.constant([[1., 1.], [2., 2.]])
tf.reduce_mean(x)  # 1.5
tf.reduce_mean(x, 0)  # [1.5, 1.5]
tf.reduce_mean(x, 1)  # [1.,  2.]
```

# tf.layers.__
dense(): adds an addition layer to your network. 

tf.layers.dense(): Creates a densely-connected layer for your neural network. 
```
# layer 1: input = s
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            # layer 2: input = e1
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')
```
Each layer adjusts the output by altering its own weights (kernel) and biases. The formula for output is
```
output = activation_function((input * weight) + bias)
```
Another way to add layers is by using tf.nn. In the case below, we are using the relu activation function.
```
# first layer. collections is used later when assign to target net
with tf.variable_scope('l1'):
  w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
  b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
  l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

# second layer. collections is used later when assign to target net
with tf.variable_scope('l2'):
  w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
  b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
  self.q_eval = tf.matmul(l1, w2) + b2
```
tf.layers.flatten(): Reshapes the tensor by flattening the input axis.
```
  x = tf.placeholder(shape=(None, 4, 4), dtype='float32')
  y = flatten(x)
  # now `y` has shape `(None, 16)`

  x = tf.placeholder(shape=(None, 3, None), dtype='float32')
  y = flatten(x)
  # now `y` has shape `(None, None)
```
tf.layers.conv2d(): A 2 dimensional layer that consists of a set of “filters”. The filters take a subset of the input data at a time, but are applied across the full input (by sweeping over the input). The operations performed by this layer are still linear/matrix multiplications, but they go through an activation function at the output, which is usually a non-linear operation.

```
tf.layers.conv2d(
    inputs,
    filters,
    activation=None,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),...
)
```
# tf.train._
Namespace that contains different optimizers for training the model. 

# tf.summary._
Summary is a special TensorBoard operation that takes in a regular tenor and outputs the summarized data to your disk (i.e. in the event file).

tf.summary.scalar(): A type of entity understood by TensorBoard. Used to record the values of a scalar tensor.

tf.summary.histogram(): A type of entities understood by TensorBoard. Used to plot histogram of all the values of a non-scalar tensor (like weight or bias matrices of a neural network).
