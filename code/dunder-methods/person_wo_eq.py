# -*- coding: utf-8 -*-
'''
>>> p1 = Person('Sam', 'Smith')
>>> p2 = Person('Sam', 'Smith')
>>> p1 == p2
False
>>> p1 is p2
False
>>> p1 == p1
True
'''
__author__ = 'gchlebus'

class Person(object):
  def __init__(self, name, surname):
    self.__name = name
    self.__surname = surname

if __name__ == '__main__':
  import doctest
  doctest.testmod()