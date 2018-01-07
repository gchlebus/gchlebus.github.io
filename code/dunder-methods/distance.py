class Distance(object):
  def __init__(self, distance_km):
    self.__distance_km = float(distance_km)

  def __format__(self, fmt_spec=''):
    '''
      >>> d = Distance(1)
      >>> print(format(d))
      1.0 km
      >>> print(format(d, 'mi'))
      0.621371 mi
      >>> print(format(d, 'y'))
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
