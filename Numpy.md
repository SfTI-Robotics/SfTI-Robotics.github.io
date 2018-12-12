# Numpy
Package for matrix creation and manipulation. It uses C over python; using low level language improves processing speed by many folds. 

```
np. = function of numpy
array. = function the generated array/matrix 
```

axis = {0,1} is often used as a parameter to determine {row, col} of functions operation.
Importing Numpy
import numpy as np
Create Matrix or Array
Create an array:
np.array(value1, value2, ...) 

Create array of zeros, ones or nothing (really small numbers):
np.zeros(size, type)
np.ones(size, type)
np.empty(size, type)

Create array with evenly spaced numbers. E.g 2, 4, 6, 8:
np.arange(start, stop, step_size)
np.linspace(start, stop, num_of_values)

Restructure array into matrix. Can be used during initialization:
array.reshape([row,col])

Create random matrix:
np.random.random((X, Y))
Matrix Properties
Check matrix dimensions:
array.ndim
array.shape

Check matrix size:
array.size

Check matrix data type:
array.dtype

Matrix Evaluation
Sum of (default all) elements:
np.sum()
np.cumsum()

Min/Max values:
np.min()
np.max()

Min/Max value indices:
np.argmin()
np.argmax()

Mean value:
np.mean()
array.mean()
array.average()

Median value:
np.median()

Difference between neighbouring elements:
np.diff()

Find positions of non-zero elements:
np.nonzero()

Standard Manipulation
Dot matrices:
np.dot()
array1.dot(array2)

Sort matrix:
np.sort()

Transpose a matrix:
np.transpose()
array.T

Clip the values:
np.clip()

Stack 2 matrix vertical or horizontally together:
np.vstack()
np.hstack()
np.newaxis()		// useful when transposing 1D matrix (array into matrix)
array.concatenate() 		// easy way to join arrays

Split a matrix into multiple matrices:
np.split()		// split into even pieces
np.array_split()	// for uneven splits. Uneven part goes to left-most matrix
np.vsplit()
