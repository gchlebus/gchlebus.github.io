---
layout: post
title: Dunder Methods
categories: python
---

The term *dunder methods* refers to Python special methods, which can make our custom defined objects take advantage of the Python language features. *Dunder* stems from the special method name spelling, which involves leading and trailing *double underscores*, e.g., `__len__()` or `__call__()`. The dunder methods are meant to be called by the Python interpreter (think of Python as a framework calling the special methods).

Let's consider an example of checking list length. We can write:
```
>>> l = 'a b c d e f'.split()
>>> len(l)
6
>>> l.__len__()
6
```

The first way is smarter, since it allows Python to make some optimizations for us. For example calling `len()` on built-in types will return the `ob_size` field in the `PyVarObject`, which is faster than calling a method.

Now, let's take a look at other special methods which can help you in making your classes truly pythonic.

### Instance creation and destruction
- `__init__(self[, ...])`: Implement here initialisation of the object after its creation.
- `__new__(cls[, ...])`: Use this static method to customize creation of an object. Normally it returns (but it doesn't have to) an instance of the `cls`.
- `__del__(self)`: Finalizer method, which gets called when the reference count of an object reaches 0. It is not guaranteed, that the function will be called for existing objects at the time Python interpreter exits. 

### String/byte representation
- `__repr__(self)`: Invoked by `repr(object)`. Used to create the *official* string representation of an object. The official representation should be of this form `<representation>`. This string is used for debugging, so make sure it contains all required informations.
- `__str__(self)`: Invoked by `str(object)`. Used to produce the *informal* string representation which is pretty and can be nicely printed. In case the method is not present, Python will call the `__repr__` method.
- `__bytes__(self)`
- `__format__(self, format_spec)`: Use this function to change the class behavior when calling `format()` or `str.format()`. For example:

```python
class Distance(object):
  def __init__(self, distance_km):
    self.__distance_km = float(distance_km)

  def __format__(self, fmt_spec=''):
    unit = fmt_spec if fmt_spec else 'km'
    if fmt_spec == 'mi':
      value = 0.621371 * self.__distance_km
    elif fmt_spec == 'y':
      value = 1093.61 * self.__distance_km
    else:
      value = self.__distance_km
    return '{} {}'.format(value, unit)
```
results in:

```
>>> print(format(Distance(1)))
1.0 km
>>> print(format(Distance(1), 'mi'))
0.621371 mi
>>> print(format(Distance(1), 'y'))
1093.61 y
```

___
### References
1. [Python Data Model](https://docs.python.org/3/reference/datamodel.html#special-method-names)
2. Luciano Ramahlho, *Fluent Python*. O'Reilly 2015.