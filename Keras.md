# Keras

# Making a  Model
keras is a simplfied version of tensorflow, although you'll need to us mainly tensorflow its a good idea to start building Neural Networks(NN) with keras as it quite readable and easy to programme. Below I'll give you a quick tutorial on how to start building you NN along with the syntax you'll need to know.

## Steps
1. Load Data

2. Define Model

3. Compile Model

4. Fit Model

5. Evaluate Model

6. Make Predictions

## Load Data 
```
numpy.random.seed(7)
```
This isn't keras specific but is always a good idea to set the random number seed when working with algorithms that use a stochastic procces. This is so that you can run the same code again and again and get the same result. This is useful if you need to demonstrate a result, compare algorithms using the same source of randomness or to debug a part of your code.


## Define Model
Models in Keras are defined as a sequence of layers.We create a Sequential model and add layers one at a time. 

_How do we know the number of layers and their types?_

This is a very hard question. Often the best network structure is found through a process of trial and error. Generally, you need a network large enough to capture the structure of the problem if that helps at all.



****************************************************************************
### `Dense()`
Fully connected layers are defined using the Dense class. 
```
model.add(Dense(12, input_dim=8, init='uniform',activation='relu'))

```
The first thing to get right is to ensure the input layer has the right number of inputs.This can be specified when creating the first layer with the input_dim argument and setting it to 8 for the 8 input variables.

The input before that _12_ is actually the number of nodes in the second layer (see diagram below for illustration)\

#### `init()`
<details><summary>Cool Dropdown #1</summary><br>


Initializations define the way to set the initial random weights of Keras layers.
  
The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply kernel_initializer and bias_initializer:
  
```
model.add(Dense(64,kernel_initializer='random_uniform',bias_initializer='zeros'))
```
  
Types:
  
```
keras.initializers.Initializer()
```
 
Initializer base class: all initializers inherit from this class.


keras.initializers.Zeros()
  
Initializer that generates tensors initialized to 0.

Ones()
  
Initializer that generates tensors initialized to 1.
  
```
keras.initializers.Constant(value=0)
```
  
Initializer that generates tensors initialized to a constant value.
  

```
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```
  
Initializer that generates tensors with a uniform distribution.

</details>


Relu stands for Rectified Linear Unit.

#### `Activation()`


 Restricts data to a rangeeg:softmax, tanh, abs, sigmoid
  

 Graphs of those functions
  

 Activation is also another function that can be called in dense or by it self
 

****************************************************************************

If you wanted to see the weights set for each layer use `layer.get_weights()`to return a list of numpy arrays.

### Creating Model

```
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

#### `Sequential()`
  
Inherits with model creates a linear stack of layers.
```
unbound_model = Sequential([l1, l2])
```

![alt text]( https://cloud.githubusercontent.com/assets/1584365/26314676/4f8eb83c-3f41-11e7-9183-2406c7a8759e.png "Logo Title Text 2")

### Types of Layers
  
Tensorflow proposes on the one hand a low level API (`tf.`, `tf.nn.`...), and on the other hand, a higher level API (`tf.layers.`, `tf.losses.`,...).
The goal of the higher level API is to provide functions that greatly simplify the design of the most common neural nets. The lower level API is there for people with special needs, or who wishes to keep a finer control of what is going on.

If you wanted something closer to tensorflow with more customisable layers use lambda.

### `Lambda()`
In Python anonymous functions are defined using the lambda keyword.
Keras employs a similar naming scheme to define anonymous/custom layers. Lambda layers in Keras help you to implement layers or functionality that is not prebuilt and which do not require trainable weights.
```
hidden_layer = lambda: Dense(num_hidden_neurons, activation=cntk.ops.relu)
keras.layers.Lambda(function, output_shape=None,mask=None,arguments=None)
```

## Compile Model

Now that the model is defined, we can compile it.

Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as TensorFlow. The backend automatically chooses the best way to represent the network for training and making predictions to run on your hardware, such as CPU or GPU or even distributed.

When compiling, we must specify some additional properties required when training the network. **Remember training a network means finding the best set of weights to make predictions for this problem.**

We must specify the loss function to use to evaluate a set of weights, the optimizer used to search through different weights for the network and any optional metrics we would like to collect and report during training.

```

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

 Another more likely loss function that you can use :

#### `reduce_mean()`
Computes the mean of elements across dimensions of a tensor. Axis input is the dimensions given
```
loss = reduce_mean(square(model - q_target), axis=0)
```

### Optimizers

Keras has a library to call different optimisers on a network
  
Importation from keras import optimizers

Implementaion example: stochastic gradient descent

``` 
learner = sgd(model.parameters,lr,gradient_clipping_threshold_per_sample=10)
```

## Fit Model

We have defined our model and compiled it ready for efficient computation.

Now it is time to execute the model on some data.

We can train or fit our model on our loaded data by calling the fit() function on the model.

The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the nepochs argument. We can also set the number of instances that are evaluated before a weight update in the network is performed, called the batch size and set using the batch_size argument.

```
model.fit(X, Y, epochs=150, batch_size=10)
```

## Evaluate Model

We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset.

This will only give us an idea of how well we have modeled the dataset (e.g. train accuracy), and how well the algorithm might perform on new data. If you separate your data into train and test datasets for training and evaluation of your model.

You can evaluate your model on your training dataset using the evaluate() function on your model and pass it the same input and output used to train the model.

This will generate a prediction for each input and output pair and collect scores, including the average loss and any metrics you have configured, such as accuracy.

```
scores = model.evaluate(X, Y)
```

If you wanted to print out your model to get an idea of what it looks like you could use `model.summary()` this is what it will look like:


![alt text]( https://i.stack.imgur.com/YbmUe.png "Logo Title Text 2")

## Make Predictions

This is a simple function that can use your trained model to make predictions on new data.

Making predictions is as easy as calling model.predict().

```
predictions = model.predict(X)
```

We are using a sigmoid activation function on the output layer(see layers above) so the predictions will be in the range between 0 and 1. We can easily convert them into a crisp binary prediction for this classification task by rounding them.



_*Congrats you have graduated from the school of keras*_











