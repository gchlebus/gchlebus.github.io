# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

from array import array

class Distance(object):
  typecode = 'd' # Array item type. Used in bytes and frombytes methods.

  def __init__(self, distance_km):
    self.__distance_km = float(distance_km)

  @classmethod
  def frombytes(cls, octets):
    '''
    >>> octets = bytes(Distance(1.2))
    >>> Distance.frombytes(octets)
    Distance(1.2)
    '''
    mview = memoryview(octets).cast(cls.typecode)
    return cls(mview[0])

  def __repr__(self):
    '''
    >>> repr(Distance(3.4))
    'Distance(3.4)'
    '''
    return '{}({})'.format(self.__class__.__name__, self.__distance_km)

  def __str__(self):
    '''
    >>> print(Distance(5.6))
    5.6 km
    '''
    return '{} km'.format(self.__distance_km)

  def __bytes__(self):
    r'''
    >>> bytes(Distance(1.0))
    b'\x00\x00\x00\x00\x00\x00\xf0?'
    '''
    return bytes(array(self.typecode, [self.__distance_km]))

  def __format__(self, fmt_spec=''):
    '''
    >>> print(format(Distance(1)))
    1.0 km
    >>> print(format(Distance(1), 'mi'))
    0.621371 mi
    >>> print(format(Distance(1), 'y'))
    1093.61 y
    '''
    unit = fmt_spec if fmt_spec else 'km'
    if fmt_spec == 'mi':
      value = 0.621371 * self.__distance_km
    elif fmt_spec == 'y':
      value = 1093.61 * self.__distance_km
    else:
      value = self.__distance_km
    return '{} {}'.format(value, unit)


if __name__ == '__main__':
  import doctest
  doctest.testmod()
