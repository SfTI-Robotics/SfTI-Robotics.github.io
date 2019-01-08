# Python

### List Notation

[None] = any input 

[ : ] = all input

[ ] = empty

### Built-in Methods

For all Built-in Methods for Python lists, see [this webpage](https://www.programiz.com/python-programming/list#built).

<details><summary> ### [`hasattr()`](https://www.programiz.com/python-programming/methods/built-in/hasattr)  </summary>
<p>

Checks if an object has an attribute by passing in the name to search for it. Returns a boolean.

Example: 
```
#----- script ---------

class Person:
    age = 23
    name = 'Adam'

person = Person()

print('Person has age?:', hasattr(person, 'age'))
print('Person has salary?:', hasattr(person, 'salary'))
#----- shell output -----

Person has age?: True
Person has salary?: False
```
</p>
</details>

### [`zip()`](https://www.programiz.com/python-programming/methods/built-in/zip)

Takes an iterator object and returns an iterator of tuple that are aggregated. 

Example: 
```
#----- script ---------
numberList = [1, 2, 3]
strList = ['one', 'two', 'three']

# No iterables are passed
result = zip()

# Converting itertor to list
resultList = list(result)
print(resultList)

# Two iterables are passed
result = zip(numberList, strList)

# Converting itertor to set
resultSet = set(result)
print(resultSet)

#----- shell output -----
[]
{(2, 'two'), (3, 'three'), (1, 'one')}
```


### Collections — High-performance container datatypes¶

##### Deque (pronouced 'deck') 
Short for double-ended queue. They are sequence (list-like) containers with dynamic sizes that can be expanded or contracted on both ends (either its front or its back).





### Other peculiar sightings 

- `\` is used when code is continued on next line (to improve readability).

### SumTree
