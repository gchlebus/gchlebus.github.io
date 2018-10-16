# -*- coding: utf-8 -*-

from entropy import *
import pytest

def test_cce():
  e = cce_loss([0.3, 0.6, 0.1], [0, 1, 0])
  assert pytest.approx(0) == e[0]
  assert pytest.approx(0.5108256) == e[1]
  assert pytest.approx(0) == e[2]

def test_bce():
  e = bce_loss([0.3, 0.6, 0.1], [0, 1, 0])
  assert pytest.approx(0.35667497) == e[0]
  assert pytest.approx(0.5108256) == e[1]
  assert pytest.approx(0.10536055) == e[2]

def test_softmax():
  o = softmax([100, 200, 1, 4])
  assert pytest.approx(1) == 1.0
