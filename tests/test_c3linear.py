import pytest

from libstatic._lib.c3linear import c3_merge, mro

class Simple(type):
    """Simplify class representation"""
    def __repr__(cls):
        return cls.__name__


class Root(object, metaclass=Simple):
    """Parent class"""
    pass


class A(Root):
    pass


class B(Root):
    pass


class C1(A, B):
    """Diamond inheritance"""
    pass


class C(Root):
    pass


class D(Root):
    pass


class E(Root):
    pass


class K1(A, B, C):
    pass


class K2(D, B, E):
    pass


class K3(D, A):
    pass


class Z(K1, K2, K3):
    """
    See hierarchy of classes here:
    https://en.wikipedia.org/wiki/C3_linearization
    """
    pass


# Some wicked cool things

class U(Root):
    pass


class T(U):
    pass


class U(T):
    pass


#
# Tests
#

@pytest.mark.parametrize('cls, eq_or_exc', [
    (A, True),
    (C1, True),
    (K1, True),
    (Z, True),
    (U, True),
])
def test_mro(cls, eq_or_exc):
    if eq_or_exc is True:
        assert mro(cls, lambda cls: cls.__bases__) == cls.mro()
    else:
        with pytest.raises(eq_or_exc):
            mro(cls, lambda cls: cls.__bases__)


@pytest.mark.parametrize('lists, result', [
    (([C], [D], [A]), [C, D, A]),
    (([A], [D], [C, A]), [D, C, A]),
    (([D, Root], [B, Root], [E, Root], [D, B, E]),  [D, B, E, Root]),
    (([C, A, Root], [D, B, Root], [C, D]), [C, A, D, B, Root]),
    (([A, C], [C, A], [C, A]), ValueError),
])
def test_merge(lists, result):
    """
    For readability c3_merge function takes in lists of str instead of lists of classes
    """
    if isinstance(result, list):
        c3_merge(*lists) == result
    else:
        with pytest.raises(result):
            c3_merge(*lists)
