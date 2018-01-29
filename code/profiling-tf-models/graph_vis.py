# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

from u_net import UNet

if __name__ == '__main__':
  model = UNet()
  model.write_graph('./graph')
  