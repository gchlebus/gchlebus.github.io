# -*- coding: utf-8 -*-
'''
>>> p1 = Person('Sam', 'Smith')
>>> p2 = Person('Sam', 'Smith')
>>> p3 = Person('Henry', 'Ford')
>>> p1 == p2
True
>>> p1 is p2
False
>>> p3 < p1
True
>>> p1 >= p3
True
'''
__author__ = 'gchlebus'

import functools
import operator

@functools.total_ordering
class Person(object):
  def __init__(self, name, surname):
    self.__name = name
    self.__surname = surname

  def __iter__(self):
    '''
    >>> for i in Person('Jon', 'Doe'): print(i)
    Jon
    Doe
    '''
    return (i for i in (self.__name, self.__surname))

  def __eq__(self, other):
    return all(a == b for a, b in zip(self, other))

  def __lt__(self, other):
    if self.__name == other.__name:
      return self.__surname < other.__surname
    return self.__name < other.__surname

  def __hash__(self):
    '''
    >>> import collections
    >>> isinstance(Person, collections.Hashable)
    True
    '''
    hashes = (hash(x) for x in self)
    return functools.reduce(operator.xor, hashes, 0)

if __name__ == '__main__':
  import doctest
  doctest.testmod()
