import doctest
import unittest
import os
from importlib import import_module
from .importlib_utils import get_submodules

import libstatic.model

class TestDoctest(unittest.TestCase):
    def test_lib_doctests(self):
        failed, _ = doctest.testmod(libstatic.model, optionflags=doctest.ELLIPSIS|doctest.IGNORE_EXCEPTION_DETAIL)
        self.assertEqual(failed, 0)

    def test_readme(self):
        failed, _ = doctest.testfile(os.path.join("..", "README.md"))
        self.assertEqual(failed, 0)
