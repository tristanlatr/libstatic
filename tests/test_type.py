# from https://github.com/orsinium-labs/astypes/blob/0.2.6/tests/test_type.py
# MIT License

#  2022 Gram

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest

from libstatic import Type


t = Type


def u(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, str):
            arg = t(arg)
        new_args.append(arg)
    return Type.Union.add_args(args=new_args)


@pytest.mark.parametrize('left, right, result', [
    # simplify the same type twice
    ('float',   'float',    'float'),
    ('str',     'str',      'str'),

    # simplify int+float
    ('float',   'int',      'float'),
    ('int',     'float',    'float'),
    ('int',     'int',      'int'),

    # simplify one unknown
    ('',        'int',      'int'),
    ('int',     '',         'int'),
    ('',        '',         ''),

    # actual unions
    ('str',     'int',      u('str', 'int')),
    ('int',     'str',      u('int', 'str')),
    ('int',     'None',     u('int', 'None')),
    ('None',    'int',      u('int', 'None')),

    # unwrap nested unions
    (u('str', 'bool'), 'int', u('str', 'bool', 'int')),
    ('int', u('str', 'bool'), u('int', 'str', 'bool')),
    (u('str', 'bool'), u('int', 'bytes'), u('str', 'bool', 'int', 'bytes')),
    (u('str', 'float'), 'int', u('str', 'float')),
    (u('float', 'str'), 'int', u('float', 'str')),
])
def test_merge(left, right, result):
    if isinstance(left, str):
        left = t(left)
    if isinstance(right, str):
        right = t(right)
    if isinstance(result, str):
        result = t(result)
    assert left.merge(right).annotation == result.annotation
    assert left.merge(right) == result


@pytest.mark.parametrize('given, expected', [
    (t('int'),                      'int'),
    (u('int', 'str'),               'int | str'),
    (t('list', args=[t('int')]),    'list[int]'),
    (t(''),                         'Any'),
])
def test_annotation(given: Type, expected: str):
    assert given.annotation == expected



@pytest.mark.parametrize('left, right, expected', [
    (t('int'), t('int'), True),
    (t('Any'), t('int'), True),
    (t('object'), t('int'), True),
    (t('float'), t('int'), True),
    (u('int', 'str'), t('int'), True),
    (u('str', 'int'), t('int'), True),

    (t('str'), t('int'), False),
    (t('int'), t('float'), False),
    (u('str', 'int'), t('bytes'), False),
])
def test_supertype_of(left: Type, right: Type, expected: bool):
    assert left.supertype_of(right) is expected
