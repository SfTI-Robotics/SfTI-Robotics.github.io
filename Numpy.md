# Numpy
Package for matrix creation and manipulation. It uses C over python; using low level language improves processing speed by many folds. 

```
np. = function of numpy
array. = function the generated array/matrix 
```

`axis = {0,1}` is often used as a parameter to determine `{row, col}` of functions operation.

## Importing Numpy
```
import numpy as np
```
## Create Matrix or Array
Create an array:
```
np.array(value1, value2, ...) 
```
Create array of zeros, ones or nothing (really small numbers):
```
np.zeros(size, type)
np.ones(size, type)
np.empty(size, type)
```

Create array with evenly spaced numbers. E.g 2, 4, 6, 8:
```
np.arange(start, stop, step_size)
np.linspace(start, stop, num_of_values)
```

Restructure array into matrix. Can be used during initialization:
```
array.reshape([row,col])
```

Create random matrix:
```
np.random.random((X, Y)) 
Matrix Properties
Check matrix dimensions:
array.ndim
array.shape
```

Check matrix size:
```
array.size
```
Check matrix data type:
```
array.dtype
```

### Matrix Evaluation
Sum of (default all) elements:
```
np.sum()
np.cumsum()
```

Min/Max values:
```
np.min()
np.max()
```

Min/Max value indices:
```
np.argmin()
np.argmax()
```

Mean value:
```
np.mean()
array.mean()
array.average()
```

Median value:
```
np.median()
```

Difference between neighbouring elements:
```
np.diff()
```

Find positions of non-zero elements:
```
np.nonzero()
```

## Standard Manipulation
Dot matrices:
```
np.dot()
array1.dot(array2)
```

Sort matrix:
```
np.sort()
```

Transpose a matrix:
```
np.transpose()
array.T
```

Clip the values:
```
np.clip()
```

Stack 2 matrix vertical or horizontally together:
```
np.vstack()
np.hstack()
np.newaxis()		// useful when transposing 1D matrix (array into matrix)
array.concatenate() 		// easy way to join arrays
```
Split a matrix into multiple matrices:
```
np.split()		// split into even pieces
np.array_split()	// for uneven splits. Uneven part goes to left-most matrix
np.vsplit()
```
## Random Functions
DISCLAMER: There are two random packages , one with numpy.random which creates a random matrix and you have a standard random package which creates a random value

Picking random numbers based of a random distribution where each value has a random chance of being selected between a minimal and a maximum: 
```
tradeoff = random.random(0,1)
```
Picking random numbers based of a uniform distribution where each value has an equal chance of selection between your minimal and maximum value
```
tradeoff = random.uniform(0,1)
```

Picking random numbers based on a normal distribution of chances for values between minimal and maximum values
```
np.random.randn(1, env.action_space.n)
```
