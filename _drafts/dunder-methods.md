---
layout: post
title: Dunder Methods
categories: python
---

The term *dunder methods* refers to Python special methods, which can make our custom defined objects take advantage of the Python language features. *Dunder* stems from the special method name spelling, which involves leading and trailing *double underscores*, e.g., `__len__()` or `__call__()`. The dunder methods are meant to be called by the Python interpreter. You can think of Python as a framework calling the special methods.

Let's consider an example of checking list length. We can write:
```python
l = 'a b c d e f'.split()
print len(l)
```
or
```python
print l.__len__()
```

The first way is smarter since it allows the Python to make some optimizations for us. For example calling `len()` on built-in types will return the `ob_size` field in the `PyVarObject`, which is faster than calling a method.

Now, let's take a look at other special methods which can help you in making your classes truly pythonic.

- `__init__(self[, ...])`: Implement here initialisation of the object after its creation.
- `__new__(cls[, ...])`: Use this static method to customize creation of an object. Normally it returns (but it doesn't have to) an instance of the `cls`.

- `__repr__(self)`: Invoked by `repr(object)`. Used to create the *official* string representation of an object. The official representation should be of this form `<representation>`. This string is used for debugging, so make sure it contains all required informations.
- `__str__(self)`: Invoked by `str(object)`. Used to produce the *informal* string representation which is pretty and can be nicely printed. In case the method is not present, Python will call the `__repr__` method.


https://docs.python.org/3/reference/datamodel.html#special-method-names