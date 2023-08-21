"""Unit tests for LazySeq.

MIT License
===========

Copyright © 2021 Claudio Jolowicz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

**The software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.**


"""

import pytest

from libstatic._lib.model import LazySeq


def test_init() -> None:
    """It is created from an iterable."""
    LazySeq([])


def test_len() -> None:
    """It returns the number of items."""
    s: LazySeq[int] = LazySeq([])
    assert 0 == len(s)


def test_getitem() -> None:
    """It returns the item at the given position."""
    s = LazySeq([1])
    assert 1 == s[0]

def test_getitem_state() -> None:
    s = LazySeq(range(15))
    assert s._curr()==-1
    assert 0 == s[0]
    assert s._curr()==0
    for i,v in zip(range(3), s):
        assert s[i] == v
        assert s._curr()==i
    assert s._curr()==2
    assert list(s) == list(LazySeq(range(15)))
    assert 4 == s[4]
    assert s._curr()==14
    assert len(s) == 15

    s = LazySeq(range(15))
    assert s._curr()==-1
    assert 0 == s[0]
    assert s._curr()==0
    assert 4 == s[4]
    assert s._curr()==4
    assert len(s) == 15
    assert list(s) == list(LazySeq(range(15)))


def test_getitem_second() -> None:
    """It returns the item at the given position."""
    s = LazySeq([1, 2])
    assert 2 == s[1]


def test_getitem_negative() -> None:
    """It returns the item at the given position."""
    s = LazySeq([1, 2])
    assert 2 == s[-1]


def test_getitem_past_cache() -> None:
    """It returns the item at the given position."""
    s = LazySeq([1, 2])
    assert (1, 2) == (s[0], s[1])


def test_getslice() -> None:
    """It returns the items at the given positions."""
    s = LazySeq([1, 2])
    [item] = s[1:]
    assert 2 == item


def test_getslice_negative_start() -> None:
    """It returns the items at the given positions."""
    s = LazySeq([1, 2])
    [item] = s[-1:]
    assert 2 == item


def test_getslice_negative_start_empty() -> None:
    """It returns the items at the given positions."""
    s: LazySeq[int] = LazySeq([])
    for _ in s[-1:]:
        pass


def test_getslice_negative_stop() -> None:
    """It returns the items at the given positions."""
    s = LazySeq([1, 2])
    [item] = s[:-1]
    assert 1 == item


def test_getslice_negative_stop_empty() -> None:
    """It returns the items at the given positions."""
    s: LazySeq[int] = LazySeq([])
    for _ in s[:-1]:
        pass


def test_getslice_negative_step() -> None:
    """It returns the items at the given positions."""
    s = LazySeq([1, 2])
    a, b = s[::-1]
    assert (2, 1) == (a, b)


def test_getslice_negative_step_and_start() -> None:
    """It returns the items at the given positions."""
    s = LazySeq([1, 2, 3])
    a, b, c = s[3::-1]
    assert (3, 2, 1) == (a, b, c)


def test_getslice_negative_step_and_stop() -> None:
    """It returns the items at the given positions."""
    s = LazySeq([1, 2, 3])
    [a] = s[:1:-1]
    assert 3 == a


def test_outofrange() -> None:
    """It raises IndexError."""
    s: LazySeq[int] = LazySeq([])
    with pytest.raises(IndexError):
        s[0]


def test_outofrange_negative() -> None:
    """It raises IndexError."""
    s = LazySeq([1, 2])
    with pytest.raises(IndexError):
        s[-3]


def test_bool_false() -> None:
    """It is False for an empty sequence."""
    s: LazySeq[int] = LazySeq([])
    assert not s


def test_bool_true() -> None:
    """It is False for a non-empty sequence."""
    s = LazySeq([1])
    assert s


def test_iter() -> None:
    """It iterates over the items in the sequence."""
    s = LazySeq([1, 2, 3])
    a, b, c = s
    assert (1, 2, 3) == (a, b, c)


def test_paging() -> None:
    """It can be used to obtain successive slices."""
    s = LazySeq(range(10000))
    while s:
        s = s[10:]
    assert not s
