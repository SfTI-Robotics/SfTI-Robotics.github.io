# Pandas
Addition package based off of numpy; Mainly used for table manipulation. i.e matrix with strings
Pandas can input and output data to other formats

```
pd. = function of pandas
df. = function of generated data frame
Importing Pandas
import pandas as pd
Create a table
Create a 1D array:
pd.Series()
```

Create a list of dates:
```
pd.date_range()
```

Create a data frame:
```
pd.DataFrame([data], index, col, dtype)
Data Frame Properties
df.index
df.columns
df.value
df.describe		// only describes numbers
```

## Basic Manipulation
Sort a data frame:
```
df.sort_index()
df.sort_value()
```

Transpose the data frame:
```
df.T
Search….
List data by row or col name:
pd.row
pd.col
```

List data of selected rows:
```
df[1, 3]
```

Select by label:
```
df.loc[rows, cols]
```

Select by position:
```
df.iloc[ ]
```

Select by both:
```
df.ix[ ]
```

...and Replace
Replace selected index:
```
df.iloc[] = 1
df.loc[] = 1
df[df.A>4] = 0 		// example
//e.g: replace all rows to 0 if the column value in that row is greater than 4
np.nan 		// if you want NaN instead
```

Add col to data frame:
```
df[‘A’] = 1		// col titled A with 1s
```

Manage NaN
Remove rows or cols with NaN in data frame:
df.dropna()

Replace NaN in data frame:
df.fillna(value)

Check NaN:
df.isnull()		// returns boolean
np.any(df.isnull()) == True	// return if any value in data frame contains NaN
  
Input and Output
Read data:
data = pd.read_’format’(‘file’)

Convert and output data to different formats:
data.to_’format’(‘new_file’)

Concatenate
pd.concat([df1. df2. df3], axis, ignore_index, join, axis, join_axes)		
// ignore_index: redo index for whole data frame
					// join: determine how data is joined {outer, inner} {fill, cut}
					// join_axes: determines if join reacts to row or col

Append data frame to the bottom of another one:
df.append()
Merge
Merge based on a key:
pd.merge(df1, df2, on = key)		
// on: merge based on properties of which key
// will discard non-matching keys

Merge based on keys:
pd.merge(df1, df2, left_index, right_index, on = ['key1', 'key2'], suffixes, how, indicator)
// how = {'outer', 'inner', 'left', 'right'} 
// indicator: indicates which df data is taken from
// left_index, right_index: which df to consider
// suffixes: merge with suffixes names

Plotting
import matplotlib.pyplot as plt

data .plot()			# all can be used with DataFrame
plt.plot()
plt.show()

plt.scatter()
data.plot.scatter()

// There is also other plot types: bar, histo, box, kde, area, hexbin, pie


