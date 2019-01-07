# Keras

# Making a  Model
keras is a simplfied version of tensorflow, although you'll need to us mainly tensorflow its a good idea to start building Neural Networks(NN) with keras as it quite readable and easy to programme. Below I'll give you a quick tutorial on how to start building you NN along with the syntax you'll need to know.

## Layers

## NN vs Layers(dense)
Tensorflow proposes on the one hand a low level API (`tf.`, `tf.nn.`...), and on the other hand, a higher level API (`tf.layers.`, `tf.losses.`,...).
The goal of the higher level API is to provide functions that greatly simplify the design of the most common neural nets. The lower level API is there for people with special needs, or who wishes to keep a finer control of what is going on.

### `Dense()`
Is a class that inherits from layers
Just your regular densely-connected NN layer.Eg:
```
model.add(Dense(12, input_dim=8, init='uniform',activation='relu'))

```
Relu stands for Rectified Linear Unit.
Activation is also another function that can be called in dense or by it self(more below)
Same with init

### `Lambda()`
In Python anonymous functions are defined using the lambda keyword.
Keras employs a similar naming scheme to define anonymous/custom layers. Lambda layers in Keras help you to implement layers or functionality that is not prebuilt and which do not require trainable weights.
```
hidden_layer = lambda: Dense(num_hidden_neurons, activation=cntk.ops.relu)
keras.layers.Lambda(function, output_shape=None,mask=None,arguments=None)
```

### `init()`
Initializations define the way to set the initial random weights of Keras layers.
The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply `kernel_initializer` and `bias_initializer`:
```
model.add(Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros'))
```
Types:
`keras.initializers.Initializer() `
Initializer base class: all initializers inherit from this class.

`keras.initializers.Zeros()`
Initializer that generates tensors initialized to 0.

`Ones()`
Initializer that generates tensors initialized to 1.

`keras.initializers.Constant(value=0)`
Initializer that generates tensors initialized to a constant value.


`keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)`
Initializer that generates tensors with a uniform distribution.

### `Activation()`
Restricts data to a rangeeg:softmax, tanh, abs, sigmoid
Graphs of those functions
How to implement

## Models
Is a class

### `Sequential()`
Inherits with model creates a linear stack of layers.
```
unbound_model = Sequential([l1, l2])
```
## Math
### `reduce_mean()`
Computes the mean of elements across dimensions of a tensor. Axis input is the dimensions given
```
loss = reduce_mean(square(model - q_target), axis=0)
```
## Optimizers
Keras has a library to call different optimisers on a network
Importation from keras import optimizers
Implementaion example: stochastic gradient descent
``` 
learner = sgd(model.parameters,lr,gradient_clipping_threshold_per_sample=10)
```





