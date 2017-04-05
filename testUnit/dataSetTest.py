import pytest
import testUnit
import preProcess.adjust as ad
import math
from numpy import *
# content of test_assert1.py




def test_function():
    assert ad.Distance((1,1), (1,1)) == 0
    assert ad.Distance((-1, 1), (0, 0)) == math.sqrt(2)
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']  # four samples and two classes
    testX = array([1.2, 1.0])
