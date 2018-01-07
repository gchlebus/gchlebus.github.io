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
- `__repr__(self)`: Invoked by `repr(object)`. Used to create the *official* string representation of an object. There is a convention, that this function should try to return a string, which when passed to `eval()` yields an object. Otherwise, a string encompassed with angle brackets with the class name and additional information should be returned, e.g., `<ClassName address>`.
- `__str__(self)`: Invoked by `str(object)`. Used to produce the *informal* string representation which is pretty and can be nicely printed. In case the method is not present, Python will call the `__repr__` method.
- `__bytes__(self)`: Used by `bytes(object)`. Return object's byte representation.
- `__format__(self, format_spec)`: Use to return a string representatino of an object depending on the format specifier string..

#### Example
```python
from array import array

class Distance(object):
  typecode = 'd' # Array item type. Used in bytes and frombytes methods.

  def __init__(self, distance_km):
    self.__distance_km = float(distance_km)

  @classmethod
  def frombytes(cls, octets):
    mview = memoryview(octets).cast(cls.typecode)
    return cls(mview[0])

  def __str__(self):
    return '{} km'.format(self.__distance_km)

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, self.__distance_km)

  def __bytes__(self):
    return bytes(array(self.typecode, [self.__distance_km]))

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
```
>>> repr(Distance(3.4))
'Distance(3.4)'
>>> print(Distance(5.6))
5.6 km
>>> print(format(Distance(1)))
1.0 km
>>> print(format(Distance(1), 'mi'))
0.621371 mi
>>> print(format(Distance(1), 'y'))
1093.61 y
>>> bytes(Distance(1.0))
b'\x00\x00\x00\x00\x00\x00\xf0?'
>>> Distance.frombytes(bytes(Distance(1.2)))
Distance(1.2)
```

### Conversion to number
- `__hash__(self)`: Called by `hash(object)`. Note, that if two objects compare equal, they have same hashes.

___
### References
1. [Python Data Model](https://docs.python.org/3/reference/datamodel.html#special-method-names)
2. Luciano Ramahlho, *Fluent Python*. O'Reilly 2015.