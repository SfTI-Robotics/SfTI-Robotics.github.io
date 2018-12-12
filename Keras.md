# Keras

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

