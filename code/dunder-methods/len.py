'''
>>> l = 'a b c d e f'.split()
>>> len(l)
6
>>> l.__len__()
6
'''
__author__ = 'gchlebus'


if __name__ == '__main__':
  import doctest
  doctest.testmod()