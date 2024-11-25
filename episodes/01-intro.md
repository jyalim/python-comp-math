---
title: Python Fundamentals
teaching: 20
exercises: 10
---

::::::::::::::::::::::::::::::::::::::: objectives

- Assign values to variables.

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::: questions

- What basic object types can I work with in Python?
- How can I create a new variable in Python?
- How do I use a function?
- Can I change the value associated with a variable after I create it?

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::  callout

If you are using a Jupyter notebook to run the examples, the keyboard
shortcut, <kbd>shift</kbd>+<kbd>enter</kbd>, will evaluate a cell and
generate output.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

## Variables

Any Python interpreter can be used as a calculator:

```python
3 + 5 * 4
```

```output
23
```


This is great but not very interesting.
To do anything useful with data, we need to assign its value to a *variable*.
In Python, we can [assign](../learners/reference.md#assign) a value to a
[variable](../learners/reference.md#variable), using the equals sign `=`.
For example, we can track the weight of a patient who weighs 60 kilograms by
assigning the value `60` to a variable `patient_weight_kg`:

```python
patient_weight_kg = 60
```

From now on, whenever we use `patient_weight_kg`, Python will substitute
the value we assigned to it. In layperson's terms, **a variable is a
name for a value**.

In Python, variable names:

- can include letters, digits, and underscores
- cannot start with a digit
- are [case sensitive](../learners/reference.md#case-sensitive).

This means that, for example:

- `weight0` and `weight_0` are a valid variable names, whereas `0weight` is not
- `weight` and `Weight` are different variables

:::::::::::::::::::::::::::::::::::::::::  callout

## Stylistic Note

Real world variables are typically given multi-word names to improve
code legibility. For instance, `patient_weight_kg` instead of just `w`
or `kg` communicates to code readers that the variable stores a weight
in kilograms for a patient. The use of underscores in the variable name
like `patient_weight_kg` is an example of *snake case*, which is the
[cultural style][PEP8-style] of Python.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

## Common object types

In Python, nearly every variable is an instance of some *class*, which
provide powerful builtin methods for transforming the underlying
*object*. We start simple, by introducing three common "variable types":

- integer numbers, which are `int` objects,
- floating point numbers, which are `float` objects,
- strings, which are immutable `str` objects.

In the example above, variable `patient_weight_kg` is an `int` object with an
integer value of `60`. It is not possible to define an `int` object's
value with anything other than an integer number.  If we want to more
precisely track the weight of our patient in kilograms, we can use a
floating point value by executing:

```python
patient_weight_kg = 60.3
```

:::::::::::::::::::::::::::::::::::::::  challenge

Why is the `float` `patient_weight_kg = 60.3` less precise than 
the `int` `patient_weight_g = 60300`?

:::::::::::::::::::::::::::::::::::::::  solution
Floating-point arithmetic results in rounding errors. The default
floating-point computer number uses 64 bits to store values, such that 1
bit stores a sign, 11 bits store an *exponent*, and the remaining 52
bits store the *significand* or *fraction*. This *double precision*
system results in only $2^{52}-1$ (roughly 4.5 quintillion) numbers
exclusively between every representable power of 2, for instance,
between 1/2 and 1, or between 1 and 2. The resulting finite rational
number system is non-associative and non-commutative.  E.g., letting
$\varepsilon=2^{-52}$, $2+\varepsilon+\varepsilon \neq \varepsilon +
\varepsilon + 2$.

In a quirk of Python, base `int` integers allow arbitrary precision. 
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

To create a string, we add single or double quotes around some text.  To
identify and track a patient throughout our study, we can assign each
person a unique identifier by storing it in a string:

```python
patient_id = '001'
```

## Using Variables in Python

Once we have data stored with variable names, we can make use of it in calculations.
We may want to store our patient's weight in pounds as well as kilograms:

```python
LB_PER_KG = 2.2
patient_weight_lb = LB_PER_KG * patient_weight_kg
```

We might decide to add a prefix to our patient identifier:

```python
patient_id = 'inflam_' + patient_id
```

## Built-in Python functions

To carry out common tasks with data and variables in Python, the
language provides us with several built-in 
[functions](../learners/reference.md#function).  
To display information to the screen, we use the `print` function:

```python
print(weight_lb)
print(patient_id)
```

```output
132.66
inflam_001
```

When we want to make use of a function, referred to as calling the
function, we follow its name by parentheses. The parentheses are
important: if you leave them off, the function doesn't actually run!
Sometimes you will include values or variables inside the parentheses
for the function to use.  In the case of `print`, we use the parentheses
to tell the function what value we want to display.  We will learn more
about how functions work and how to create our own in later episodes.

We can display multiple things at once using only one `print` call:

```python
print(patient_id, 'weight in kilograms:', patient_weight_kg)
```

```output
inflam_001 weight in kilograms: 60.3
```

We can also call a function inside of another
[function call](../learners/reference.md#function-call).
For example, Python has a built-in function called `type` that tells you a value's data type:

```python
print(type(60.3))
print(type(patient_id))
```

```output
<class 'float'>
<class 'str'>
```

Moreover, we can do arithmetic with variables right inside the `print` function:

```python
print('patient weight in pounds:', 2.2 * patient_weight_kg)
```

```output
patient weight in pounds: 132.66
```

The above command, however, did not change the value of `patient_weight_kg`:

```python
print(patient_weight_kg)
```

```output
60.3
```

To change the value of the `patient_weight_kg` variable, we have to
**assign** `patient_weight_kg` a new value using the equals `=` sign:

```python
patient_weight_kg = 65.0
print('patient weight in kilograms is now:', patient_weight_kg)
```

```output
patient weight in kilograms is now: 65.0
```

:::::::::::::::::::::::::::::::::::::::::  callout

## Variables as Sticky Notes

A variable in Python is analogous to a sticky note with a name written
on it: assigning a value to a variable is like putting that sticky note
on a particular value.

![](fig/python-sticky-note-variables-01.svg){alt='Value of 65.0 with weight\_kg label stuck on it'}

Using this analogy, we can investigate how assigning a value to one
variable does **not** change values of other, seemingly related,
variables.  For example, let's store the subject's weight in pounds in
its own variable:

```python
# There are 2.2 pounds per kilogram
LB_PER_KG = 2.2
patient_weight_lb = LB_PER_KG * patient weight_kg
print('patient weight in kilograms:', patient_weight_kg, 'and in pounds:', patient_weight_lb)
```

```output
patient weight in kilograms: 65.0 and in pounds: 143.0
```

Everything in a line of code following the '#' symbol is a
[comment](../learners/reference.md#comment) that is ignored by Python.
Comments allow programmers to leave explanatory notes for other
programmers or their future selves.

![](fig/python-sticky-note-variables-02.svg){alt='Value of 65.0 with weight\_kg label stuck on it, and value of 143.0 with weight\_lb label stuck on it'}

Similar to above, the expression 
`LB_PER_KG * patient_weight_kg` 
is evaluated to
`143.0`, and then this value is assigned to the variable `patient_weight_lb`
(i.e., the sticky note `patient_weight_lb` is placed on `143.0`). 
At this point, each variable is "stuck" to completely distinct and
unrelated values.

Let's now change `patient_weight_kg` and introduce an f-string (string
with first quote prefixed with an `f`), a fast way for Python and the
programmer to interpolate strings with potentially formatted variable
values:

```python
patient_weight_kg = 100.0
print(f'patient weight in kilograms is now: {patient_weight_kg} and weight in pounds is still {patient_weight_lb}')
```

```output
patient weight in kilograms is now: 100.0 and weight in pounds is still: 143.0
```

![](fig/python-sticky-note-variables-03.svg){alt='Value of 100.0 with label weight\_kg stuck on it, and value of 143.0 with label weight\_lbstuck on it'}

Since `patient_weight_lb` doesn't "remember" where its value comes from,
it is not updated when we change `patient_weight_kg`.


::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Check Your Understanding

What values do the variables `mass` and `age` have after each of the following statements?
Test your answer by executing the lines.

```python
mass = 47.5
age = 122
mass = mass * 2.0
age = age - 20
```

:::::::::::::::  solution

## Solution

```output
`mass` holds a value of 47.5, `age` does not exist
`mass` still holds a value of 47.5, `age` holds a value of 122
`mass` now has a value of 95.0, `age`'s value is still 122
`mass` still has a value of 95.0, `age` now holds 102
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Sorting Out References

Python allows you to assign multiple values to multiple variables in one
line by separating the variables and values with commas. This kind of
syntax is called, *multiple assignment*. What does the following program
print out?

```python
first, second = 'Grace', 'Hopper'
third, fourth = second, first
print(third, fourth)
a, b = d, c = 'Emmy', 'Noether'
print(c,d)
```

:::::::::::::::  solution

## Solution

```output
Hopper Grace
Noether Emmy
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::  challenge

## Seeing Object Types

What are the object types of the following variables?

```python
planet = 'Earth'
apples = 5
distance = 10.5
```

:::::::::::::::  solution

## Solution

```python
print(type(planet))
print(type(apples))
print(type(distance))
```

```output
<class 'str'>
<class 'int'>
<class 'float'>
```

:::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::::::::::::::::

## Basic arithmetic operations

In Python, exponentiation of `int` and `float` objects is done with the
`**` operator. 

```python
print((1+1e-3)**1000)
```
```output
2.7169239322355936
```

Something quirky is that `/` is a *true divide* whereas `//` does an
integer divide. 

```python
print(4/3,4//3,4e0//3e0)
```
```output
1.3333333333333333 1 1.0
```

The `%` computes a modulus

```python
e_7 = (1+1e-7)**1e7
print(e_7%2,12%7)
```
```output
0.7182816941320818 5
```

Since most variables are objects in python, all operators may be
redefined, but this is generally not recommended. Instead, operators can
be leveraged to provide greater high-level functionality. For instance,
the builtin `str` class uses "multiplication" to quickly generate a
repeated string pattern,
```python
print(36*'=-')
```
```output
=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
```
However, division is not defined.
```python
print('this will fail'/1)
```
```output
TypeError: unsupported operand type(s) for /: 'str' and 'int'
```

A final operator to note is the `matmul` symbol, `@`. This will be
utilized in the next section after introducing a powerful Python library
called `numpy`.

:::::::::::::::::::::::::::::::::::::::: keypoints

- Basic data types in Python include integers, strings, and floating-point numbers.
- Use `variable = value` to assign a value to a variable in order to record it in memory.
- Variables are created on demand whenever a value is assigned to them.
- Use `print(something)` to display the value of `something`.
- Use `# some kind of explanation` to add comments to programs.
- Built-in functions are always available to use.

::::::::::::::::::::::::::::::::::::::::::::::::::


[PEP8-style]: https://peps.python.org/pep-0008
