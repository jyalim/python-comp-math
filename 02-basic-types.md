---
title: Basic Python Types and Data Structures
teaching: 30
exercises: 15
---

::::::::::::::::::::::::::::::::::::::: objectives

- Understand the overview of basic Python types for working with multiple values.
- Understand the difference between mutable and immutable types.
- Explain what a list is.
- Create and index lists of simple values.
- Change the values of individual elements
- Append values to an existing list
- Reorder and slice list elements
- Create and manipulate nested lists
- Explain what a dict is.
- Create and index dicts of simple values.
- Change the values of individual elements.
- Understand the differences between lists, tuples, sets, and dicts.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- How can I store many values together?
- What is the major difference between a list and a tuple?
- What is the major difference between a list and a dict?

::::::::::::::::::::::::::::::::::::::::::::::::::

In the previous lesson, we learned how to assign variable **names** to
single `int`s, `float`s, as well as `str`ings.

Our goal now is to introduce the basic types that Python provides for
working with multiple values under a single name. The additional
built-in types that we will use after this lesson are [lists][list-docs]
(simple object containers that would typically be called arrays in other
languages) and [dictionaries][dict-docs] (associative arrays with
arbitrary key--value mappings, type `dict`), so most of the focus will
be on them.  There are additional built-in types but giving them a full
treatment is out-of-scope for this tutorial.  For now, just note that
`list`s and `dict`s are mutable objects: elements of either can be
arbitrarily changed in place. The `tuple` is identical to a `list`
except that it is immutable: attempting to change a value in a `tuple`
will throw an error. This is also true for `set`s and strings.
Additional details and examples are given in an Appendix, for instance,
the jupyter notebook `A1-basic-types.ipynb`. 

## Python lists

Lists are one of two major workhorses in Python codes for easily
collecting multiple values under a single variable name (the other being
dictionaries, which we will get to later). Lists are capable of
containing all other objects as elements, including nested lists (this
will be demonstrated later).

We create our first list by explicitly declaring its comma-separated
contents within square brackets:

```python
odds = [1, 3, 5, 7]
print(f'first {len(odds)} odds are: {odds}')
```

```output
first 4 odds are: [1, 3, 5, 7]
```

Notice that we can obtain the number of elements in the list with the
built-in function, `len`. To actually access list elements,
we can use *indices* --- sequentially numbered positions of the values
in the list.  **Python is zero-indexed: these positions are numbered
starting at 0, and the first element has an index of 0.** 

```python
print('first element:', odds[0])
print('last element:', odds[3])
print('last element:', odds[len(odds)-1])
print('"-1" element:', odds[-1])
```

```output
first element: 1
last element: 7
last element: 7
"-1" element: 7
```

Negative numbers are useful ways to obtain list values and --- because
Python is zero-indexed --- are like implicit arithmetic references to
the length of the list.  When we use negative indices, the index `-1`
gives us the last element in the list, `-2` the second to last, and so
on.  Because of this, `odds[3]` and `odds[len(odds)-1]` and `odds[-1]`
point to the same element here. Below is a map of the indices that will
dereference the values of `odds` (using a block string).

```python
print("""
        +---+---+---+---+
values: | 1 | 3 | 5 | 7 |
        +---+---+---+---+
+index:   0   1   2   3 
-index:  -4  -3  -2  -1
""")
```

```output
        +---+---+---+---+
values: | 1 | 3 | 5 | 7 |
        +---+---+---+---+
+index:   0   1   2   3 
-index:  -4  -3  -2  -1
```

---

There is one important difference between lists and strings: we can
change the values in a list, but we cannot change individual characters
in a string.  For example:

```python
# typo in Darwin's name
names = ['Noether', 'Darwing', 'Turing', 'Hopper']
print('names is originally:', names)
# correct the name
names[1] = 'Darwin'  
print('final value of names:', names)
```

```output
names is originally: ['Noether', 'Darwing', 'Turing', 'Hopper']
final value of names: ['Noether', 'Darwin', 'Turing', 'Hopper']
```

works, but:

```python
name = 'Darwin'
name[0] = 'd'
```

```error
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-8-220df48aeb2e> in <module>()
      1 name = 'Darwin'
----> 2 name[0] = 'd'

TypeError: 'str' object does not support item assignment
```

does not.

:::::::::::::::::::::::::::::::::::::::::  callout

## Ch-Ch-Ch-Ch-Changes

Data which can be modified in place is called
[mutable](../learners/reference.md#mutable), while data which cannot be
modified is called [immutable](../learners/reference.md#immutable).
Strings and numbers are immutable. This does not mean that variables
with string or number values are constants, but when we want to change
the value of a string or number variable, we can only replace the old
value with a completely new value.

Lists and dictionaries, on the other hand, are mutable: we can modify
them after they have been created. We can change individual elements,
append new elements, or reorder the whole list. For some operations,
like sorting, we can choose whether to use a function that modifies the
data in-place or a function that returns a modified copy and leaves the
original unchanged.

Be careful when modifying data in-place. If two variables refer to the
same list, and you modify the list value, it will change for both
variables!

```python
mild_salsa = ['peppers', 'onions', 'cilantro', 'tomatoes']
# mild_salsa and hot_salsa point to the *same* list data in memory
hot_salsa = mild_salsa 
hot_salsa[0] = 'hot peppers'
print('Ingredients in mild salsa:', mild_salsa)
print('Ingredients in hot salsa:', hot_salsa)
```

```output
Ingredients in mild salsa: ['hot peppers', 'onions', 'cilantro', 'tomatoes']
Ingredients in hot salsa: ['hot peppers', 'onions', 'cilantro', 'tomatoes']
```

If you want variables with mutable values to be independent, you
must make a copy of the value when you assign it.

```python
import copy
mild_salsa = ['peppers', 'onions', 'cilantro', 'tomatoes']
# forces a *copy* of the list
hot_salsa = copy.deepcopy(mild_salsa)
hot_salsa[0] = 'hot peppers'
print('Ingredients in mild salsa:', mild_salsa)
print('Ingredients in hot salsa:', hot_salsa)
```

```output
Ingredients in mild salsa: ['peppers', 'onions', 'cilantro', 'tomatoes']
Ingredients in hot salsa: ['hot peppers', 'onions', 'cilantro', 'tomatoes']
```

Because of pitfalls like this, code which modifies data in place can be
more difficult to understand. However, it is often far more efficient to
modify a large data structure in place than to create a modified copy
for every small change. You should consider both of these aspects when
writing your code.


::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::  callout

## Nested Lists

Since a list can contain any Python object, it can even contain other
lists.  For example, you could represent the products on the shelves of
a small grocery shop as a nested list called `veg`:

![](fig/04_groceries_veg.png){alt='veg is represented as a shelf full of
produce. There are three rows of vegetables on the shelf, and each row
contains three baskets of vegetables. We can label each basket according
to the type of vegetable it contains, so the top row contains (from left
to right) lettuce, lettuce, and peppers.'}

To store the contents of the shelf in a nested list, you write it this
way:

```python
veg = [
  ['lettuce', 'lettuce', 'peppers', 'zucchini'],
  ['lettuce', 'lettuce', 'peppers', 'zucchini'],
  ['lettuce', 'cilantro', 'peppers', 'zucchini']
]
```

Here are some visual examples of how indexing a list of lists `veg`
works. First, you can reference each row on the shelf as a separate
list. For example, `veg[2]` represents the bottom row, which is a list
of the baskets in that row.

![](fig/04_groceries_veg0.png){alt='veg is now shown as a list of three
rows, with veg\[0\] representing the top row of three baskets, veg\[1\]
representing the second row, and veg\[2\] representing the bottom row.'}

Index operations using the image would work like this:

```python
print(veg[2])
```

```output
['lettuce', 'cilantro', 'peppers', 'zucchini']
```

```python
print(veg[0])
```

```output
['lettuce', 'lettuce', 'peppers', 'zucchini']
```

To reference a specific basket on a specific shelf, you use two indexes.
The first index represents the row (from top to bottom) and the second
index represents the specific basket (from left to right). For instance,
the cilantro is in the last row, second column, `veg[-1][1]`.

```python
print(veg[-1][1])
```

```output
'cilantro'
```


![](fig/04_groceries_veg00.png){alt='veg is now shown as a
two-dimensional grid, with each basket labeled according to its index in
the nested list. The first index is the row number and the second index
is the basket number, so veg\[1\]\[3\] represents the basket on the far
right side of the second row (basket 4 on row 2): zucchini'}

```python
print(veg[0][0])
```

```output
'lettuce'
```

```python
print(veg[1][2])
```

```output
'peppers'
```

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::  callout

## Heterogeneous Lists

Lists in Python can contain elements of different types. Example:

```python
sample_ages = [10, 12.5, 'Unknown']
```

::::::::::::::::::::::::::::::::::::::::::::::::::

There are many ways to change the contents of lists besides assigning
new values to individual elements:

```python
odds.append(11)
print('`odds` after adding a value:', odds)
```

```output
`odds` after adding a value: [1, 3, 5, 7, 11]
```

```python
removed_element = odds.pop(0)
print('odds after removing the first element:', odds)
print('removed_element:', removed_element)
```

```output
odds after removing the first element: [3, 5, 7, 11]
removed_element: 1
```

```python
odds.reverse()
print('odds after reversing:', odds)
```

```output
odds after reversing: [11, 7, 5, 3]
```

While modifying in place, it is useful to remember that Python treats
lists in a slightly counter-intuitive way.

As we saw earlier, when we modified the `mild_salsa` list item in-place,
if we make a list, (attempt to) copy it and then modify this list, we
can cause all sorts of trouble. This also applies to modifying the list
using the above functions:

```python
odds = [3, 5, 7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

```output
primes: [3, 5, 7, 2]
odds: [3, 5, 7, 2]
```

This is because Python stores a list in memory, and then can use
multiple names to refer to the same list. If all we want to do is copy a
(simple) list, we can again use the `deepcopy` method from the `copy`
built-in library, so we do not modify a list we did not mean to:

```python
import copy
odds = [3, 5, 7]
primes = copy.deepcopy(odds)
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

```output
primes: [3, 5, 7, 2]
odds: [3, 5, 7]
```

Subsets of lists and strings can be accessed by specifying ranges of
values in brackets.  This is commonly referred to as "slicing" the
list/string.

```python
binomial_name = 'Drosophila melanogaster'
group = binomial_name[0:10]
print(f'group: {group}')

species = binomial_name[11:23]
print(f'species: {species}')

# using built-in string methods:
# the split method splits a string into a list wherever a blank space
# occurs (by default)
group,species = binomial_name.split()
# \n is interpreted as the newline character
print(f'group: {group}\nspecies: {species}')

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes[2:5]
print(f'autosomes: {autosomes}')

last = chromosomes[-1]
print('last:', last)
```

```output
group: Drosophila
species: melanogaster
group: Drosophila
species: melanogaster
autosomes: ['2', '3', '4']
last: 4
```

:::::::::::::::::::::::::::::::::::::::  challenge

## Slicing From the End

Use slicing to access only the last four characters of a string or
entries of a list.

```python
string_for_slicing = 'Observation date: 02-Feb-2013'
list_for_slicing = [
  ['fluorine', 'F'],
  ['chlorine', 'Cl'],
  ['bromine', 'Br'],
  ['iodine', 'I'],
  ['astatine', 'At'],
]
```

```output
'2013'
[['chlorine', 'Cl'], ['bromine', 'Br'], ['iodine', 'I'], ['astatine', 'At']]
```

Would your solution work regardless of whether you knew beforehand the
length of the string or list (e.g. if you wanted to apply the solution
to a set of lists of different lengths)?  If not, try to change your
approach to make it more robust.

Hint: Remember that indices can be negative as well as positive

:::::::::::::::  solution

## Solution

Use negative indices to count elements from the end of a container (such
as list or string):

```python
string_for_slicing[-4:]
list_for_slicing[-4:]
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Non-Contiguous Slices

So far we've seen how to use slicing to take single blocks of successive
entries from a sequence.  But what if we want to take a subset of
entries that aren't next to each other in the sequence?

You can achieve this by providing a third argument to the range within
the brackets, called the ***step size***.  The example below shows how
you can take every third entry in a list:

```python
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
subset = primes[0:12:3]
print('subset', subset)
```

```output
subset [2, 7, 17, 29]
```

Notice that the slice taken begins with the first entry in the range,
followed by entries taken at equally-spaced intervals (the steps)
thereafter.  If you wanted to begin the subset with the third entry,
you would need to specify that as the starting point of the sliced range:

```python
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
subset = primes[2:12:3]
print('subset', subset)
```

```output
subset [5, 13, 23, 37]
```

Use the step size argument to create a new string that contains only
every other character in the string "In an octopus's garden in the
shade". Start with creating a variable to hold the string:

```python
beatles = "In an octopus's garden in the shade"
```

What slice of `beatles` will produce the following output (i.e., the
first character, third character, and every other character through the
end of the string)?

```output
I notpssgre ntesae
```

:::::::::::::::  solution

## Solution

To obtain every other character you need to provide a slice with the
step size of 2:

```python
beatles[0:35:2]
```

You can also leave out the beginning and end of the slice to take the
whole string and provide only the step argument to go every second
element:

```python
beatles[::2]
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

If you want to take a slice from the beginning of a sequence, you can
omit the first index in the range:

```python
date = 'Monday 4 January 2016'
day = date[0:6]
print('Using 0 to begin range:', day)
day = date[:6]
print('Omitting beginning index:', day)
```

```output
Using 0 to begin range: Monday
Omitting beginning index: Monday
```

And similarly, you can omit the ending index in the range to take a
slice to the very end of the sequence:

```python
# These could all be set on one-line, but "exploding" improves
# readability and commentability
months = [
  'jan', 
  'feb', 
  'mar', 
  'apr', 
  'may', 
  'jun', 
  'jul', 
  'aug', 
  'sep', 
  'oct', 
  'nov', 
  'dec',
]
sond = months[8:12]
print('With known last position:', sond)
sond = months[8:len(months)]
print('Using len() to get last entry:', sond)
sond = months[8:]
print('Omitting ending index:', sond)
```

```output
With known last position: ['sep', 'oct', 'nov', 'dec']
Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']
Omitting ending index: ['sep', 'oct', 'nov', 'dec']
```

:::::::::::::::::::::::::::::::::::::::  challenge

## Overloading

`+` usually means addition, but when used on strings or lists, it means
"concatenate."  Given that, what do you think the multiplication
operator `*` does on lists?  In particular, what will be the output of
the following code?

```python
counts = [2, 4, 6, 8, 10]
repeats = counts * 2
print(repeats)
```

1. `[2, 4, 6, 8, 10, 2, 4, 6, 8, 10]`
2. `[4, 8, 12, 16, 20]`
3. `[[2, 4, 6, 8, 10], [2, 4, 6, 8, 10]]`
4. `[2, 4, 6, 8, 10, 4, 8, 12, 16, 20]`

The technical term for this is *operator overloading*: a single
operator, like `+` or `*`, can do different things depending on what
it's applied to.

:::::::::::::::  solution

## Solution

The multiplication operator `*` used on a list replicates elements of
the list and concatenates them together:

```output
[2, 4, 6, 8, 10, 2, 4, 6, 8, 10]
```

It's equivalent to:

```python
counts + counts
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

## Dictionaries

Dictionaries are the second of two major workhorses in Python codes for
easily collecting multiple values under a single variable name.
Like lists, dictionary values are capable of containing all other
objects as elements, including nested dictionaries or lists. However,
unlike lists --- which always use integers starting from `0` elements
--- dictionaries allow for arbitrary indices, called ***keys***, that
must simply be immutable. This means that a dictionary or list cannot be
a key, but integers, floats, complex floats, tuples, sets, and
especially strings may be.

We create our first dictionary by explicitly declaring its
***key-value*** pairs with colons and comma-separated elements within
***curly*** brackets:

```python
squares = {1:1, 2:4, 3:9, 4:16, 'five':'twenty-five'}
print(squares)
print(squares[1], squares[4], squares['five'])
```

```output
{1: 1, 2: 4, 3: 9, 4: 16, 'five': 'twenty-five'}}
1 16 twenty-five 100
```

Once a dictionary is created, new key-value pairs can be appended by
associating a new value to a new key:

```python
squares[10] = 100
print(squares)
```

```output
{1: 1, 2: 4, 3: 9, 4: 16, 'five': 'twenty-five', 10: 100}
```

Since Python 3.9, two dictionaries may be merged with the `|` operator,

```python
more_squares = {6:36, 7:49}
squares = squares | more_squares
print(squares)
```

```output
{1: 1, 2: 4, 3: 9, 4: 16, 'five': 'twenty-five', 10: 100, 6: 36, 7: 49}
```

There are two ways to remove a key-value pair:
```python
# del is a built-in statement for deleting workspace objects
del squares['five']
# or
removed_value = squares.pop(10)
```

Trying to access a dictionary via an undefined key-value pair will throw
a `KeyError` exception:

```python
print(squares[8])
```

```output
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
----> 1 print(squares[8])

KeyError: 8
```

Another common way to create dictionaries involves the constructor
function, `dict()`, but this will only work for `str`-type keys:

```python
periodic_table = dict(
  Hydrogen = 'H',
  Helium   = 'He',
  Lithium  = 'Li',
  Beryllium= 'Be',
  Boron    = 'B',
  Carbon   = 'C',
  Nitrogen = 'N',
  Oxygen   = 'O',
  Fluorine = 'F',
  Neon     = 'Ne',
)
print(periodic_table)
```

```output
{'Hydrogen': 'H', 'Helium': 'He', 'Lithium': 'Li', 'Beryllium': 'Be',
'Boron': 'B', ' Carbon': 'C', 'Nitrogen': 'N', 'Oxygen': 'O',
'Fluorine': 'F', 'Neon': 'Ne'}
```

The advantage of this alternative is that it's more transparent to the
layperson (`dict(...)` vs. `{...}`) and if a dictionary with pure string
keys needs to be created, the lack of quotes on the key names saves the
programmer time.

When the programmer wants to build containers of values with more
explicit or meaningful mappings than sequences of natural numbers,
dictionaries provide an invaluable data structure. Any time an
application accesses multiple lists simultaneously, consider whether a
dictionary would improve the readability of your code.

:::::::::::::::::::::::::::::::::::::::  challenge

## Dictionary operations

Consider the following two dictionaries.

```python
APM_Fall23_grad_courses = {
  501 : 'ODEs',
  503 : 'Analysis',
  505 : 'Linear Algebra',
}
MAT_Fall23_grad_courses = {
  501 : 'Topology',
  512 : 'Combinatorics',
  516 : 'Graph Theory',
}
```

Determine the outcome of the following code:

```python
grad_courses = APM_Fall23_grad_courses | MAT_Fall23_grad_courses
```

What if you swap the operands on either side of the ***pipe*** `|`?

What would be a better data structure convention to prevent loss of
information after the use of `|`?

:::::::::::::::  solution

## Solution

The two dictionaries will be merged into a new dictionary object,
`grad_courses`, that will contain five key-value pairs. The collision of
the `501` key is resolved by taking the right-side value. So the key
`501` will be set to the value of `Topology`. When the operands are
swapped, the value is instead set to `ODEs`.

A simple fix would have been to make the dictionary keys richer, i.e.,
`APM501` instead of `501`.

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

## Conclusion

The next lesson gets into loops, which we will quickly learn are capable
of iterating over the items of a list or dictionary. This rich
functionality will guide your non-numeric data structures when
programming with Python.




:::::::::::::::::::::::::::::::::::::::: keypoints

- `[value1, value2, value3, ...]` creates a list.
- `{key1:value1, key2:value2, ...}` creates a dictionary.
- Dictionary keys have to be immutable objects, like `int`s, `float`s,
  but especially `str`s.
- Lists and dictionaries values may be any Python object, including
  themselves (i.e., list of lists or dictionaries of dictionaries).
- Lists are indexed and sliced with square brackets (e.g., `list[0]` and
  `list[2:9]`), in the same way as strings.
- Dictionaries are indexed with square brackets too (e.g.,
  `dict['Neon']`).
- Lists and dictionaries are mutable (i.e., their values can be changed
  in place).
- Strings are immutable (i.e., the characters in them cannot be
  changed).
::::::::::::::::::::::::::::::::::::::::::::::::::


[list-docs]: https://docs.python.org/3/glossary.html#term-list
[dict-docs]: https://docs.python.org/3/glossary.html#term-dictionary
