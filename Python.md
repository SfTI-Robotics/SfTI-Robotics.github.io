# Python

If you would like a great introductory Python guide, check out [Daniel Seita's post](https://danieltakeshi.github.io/2013/07/05/ten-things-python-programmers-should-know/)

### List Notation

[None] = any input 

[ : ] = all input

[ ] = empty

### Built-in Methods

For all Built-in Methods for Python lists, see [this webpage](https://www.programiz.com/python-programming/list#built).

### [`hasattr()`](https://www.programiz.com/python-programming/methods/built-in/hasattr) 
<details> 
   <summary>  </summary>
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
<details> 
   <summary>  </summary>
<p>
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
</p>
</details>

### Collections — High-performance container datatypes¶

##### Deque (pronouced 'deck') 
Short for double-ended queue. They are sequence (list-like) containers with dynamic sizes that can be expanded or contracted on both ends (either its front or its back).



### Argparse

command line user input arguments


### Emumerate

```
for counter, value in enumerate(some_list):
    print(counter, value)
--------

my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list, 1):
    print(c, value)

# Output:
# 1 apple
# 2 banana
# 3 grapes
# 4 pear
```

- allows us to loop over something and have an automatic counter
- accepts an optional argument which specifies the starting index
- 


### Other peculiar sightings 

- `\` is used when code is continued on next line (to improve readability).

### SumTree

