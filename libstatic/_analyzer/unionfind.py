# MIT License

# Copyright (c) 2017 Debajyoti Nandi
# Copyright (c) 2021 Hagai Helman Tov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
UnionFind Implementation in Python

>>> uf = UnionFind(list('abcdefghij'))
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
par=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
n_elts=10,n_comps=10>
>>> uf.union('e', 'd')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
par=[0, 1, 2, 4, 4, 5, 6, 7, 8, 9],
n_elts=10,n_comps=9>
>>> uf.union('d', 'i')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 3, 1, 1, 1, 1, 1],
par=[0, 1, 2, 4, 4, 5, 6, 7, 4, 9],
n_elts=10,n_comps=8>
>>> uf.union('g', 'f')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 3, 1, 2, 1, 1, 1],
par=[0, 1, 2, 4, 4, 6, 6, 7, 4, 9],
n_elts=10,n_comps=7>
>>> uf.union('j', 'e')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 4, 1, 2, 1, 1, 1],
par=[0, 1, 2, 4, 4, 6, 6, 7, 4, 4],
n_elts=10,n_comps=6>
>>> uf.union('c', 'b')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 2, 1, 4, 1, 2, 1, 1, 1],
par=[0, 2, 2, 4, 4, 6, 6, 7, 4, 4],
n_elts=10,n_comps=5>
>>> uf.union('i', 'j')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 2, 1, 4, 1, 2, 1, 1, 1],
par=[0, 2, 2, 4, 4, 6, 6, 7, 4, 4],
n_elts=10,n_comps=5>
>>> uf.union('f', 'a')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 2, 1, 4, 1, 3, 1, 1, 1],
par=[6, 2, 2, 4, 4, 6, 6, 7, 4, 4],
n_elts=10,n_comps=4>
>>> uf.union('h', 'c')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 3, 1, 1, 1],
par=[6, 2, 2, 4, 4, 6, 6, 2, 4, 4],
n_elts=10,n_comps=3>
>>> uf.union('g', 'b')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 2, 6, 4, 4, 6, 6, 2, 4, 4],
n_elts=10,n_comps=2>
>>> uf.union('a', 'b')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 6, 6, 4, 4, 6, 6, 2, 4, 4],
n_elts=10,n_comps=2>
>>> uf.union('g', 'h')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 6, 6, 4, 4, 6, 6, 6, 4, 4],
n_elts=10,n_comps=2>
>>> uf.connected('a', 'g')
True
>>> uf.component('a')
<ordered_set {g, a, b, c, f, h}>
>>> uf.components()
[<ordered_set {g, a, b, c, f, h}>, <ordered_set {e, d, i, j}>]
"""
from __future__ import annotations
from collections import deque

from typing import (
    Iterable,
    Iterator,
    Hashable,
    TypeVar,
    Generic, 
    Collection,
    TYPE_CHECKING
)
from beniget.beniget import ordered_set

T = TypeVar('T', bound=Hashable)

class UnionFind(Generic[T], Collection[T]):
    """
    Union-find disjoint sets datastructure.

    Union-find is a data structure that maintains disjoint set
    (called connected components or components in short) membership,
    and makes it easier to merge (union) two components, and to find
    if two elements are connected (i.e., belong to the same
    component).

    This implements the "weighted-quick-union-with-path-compression"
    union-find algorithm.  Only works if elements are immutable
    objects.

    Worst case for union and find: :math:`(N + M \log^* N)`, with
    :math:`N` elements and :math:`M` unions. The function
    :math:`\log^*` is the number of times needed to take :math:`\log`
    of a number until reaching 1. In practice, the amortized cost of
    each operation is nearly linear [1]_.

    Terms
    -----
    Component
        Elements belonging to the same disjoint set

    Connected
        Two elements are connected if they belong to the same component.

    Union
        The operation where two components are merged into one.

    Root
        An internal representative of a disjoint set.

    Find
        The operation to find the root of a disjoint set.

    Parameters
    ----------
    elements : NoneType or container, optional, default: None
        The initial list of elements.

    Attributes
    ----------
    n_elts : int
        Number of elements.

    n_comps : int
        Number of distjoint sets or components.

    .. [1] http://algs4.cs.princeton.edu/lectures/

    """

    def __init__(self, elements:Iterable[T]|None=None) -> None:
        self.n_elts = 0  # current num of elements
        self.n_comps = 0  # the number of disjoint sets or components
        self._next = 0  # next available id
        self._elts: list[T | None] = []  # the elements, or None for empty spots
        self._indx: dict[T, int] = {}  #  dict mapping elt -> index in _elts
        self._par: list[int] = []  # parent: for the internal tree structure
        self._siz: list[int] = []  # size of the component - correct only for roots
        self._removed: list[int] # indexes free to use after removals.

        if elements is None:
            elements = []
        for elt in elements:
            self.add(elt)


    def __repr__(self) -> str:
        return  (
            '<UnionFind:\nelts={},\nsiz={},\npar={},\nn_elts={},n_comps={}>'
            .format(
                self._elts,
                self._siz,
                self._par,
                self.n_elts,
                self.n_comps,
            ))

    def __len__(self) -> int:
        return self.n_elts

    def __contains__(self, x) -> bool:
        return x in self._indx

    def __getitem__(self, index:int) -> T:
        if index < 0 or index >= self._next:
            raise IndexError('index {} is out of bound'.format(index))
        return self._elts[index]
    
    def __iter__(self) -> Iterator[T]:
        return iter(self._elts)

    def add(self, x: T) -> None:
        """
        Add a single disjoint element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        None

        """
        if x in self:
            return
        self._elts.append(x)
        self._indx[x] = self._next
        self._par.append(self._next)
        self._siz.append(1)
        self._next += 1
        self.n_elts += 1
        self.n_comps += 1

    def find(self, x: T) -> int:
        """
        Find the index of root of the disjoint 
        set containing the given element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        int
            The (index of the) root.

        Raises
        ------
        ValueError
            If the given element is not found.

        """
        if x not in self._indx:
            raise ValueError('{} is not an element'.format(x))

        p = self._indx[x]
        while p != self._par[p]:
            # path compression
            q = self._par[p]
            self._par[p] = self._par[q]
            p = q
        return p

    def connected(self, x: T, y: T) -> bool:
        """Return whether the two given elements belong to the same component.

        Parameters
        ----------
        x : immutable object
        y : immutable object

        Returns
        -------
        bool
            True if x and y are connected, false otherwise.

        """
        return self.find(x) == self.find(y)

    def union(self, x: T, y: T) -> None:
        """
        Merge the components of the two given elements into one.
        Initialize elements if they are not already in the collection.

        Parameters
        ----------
        x : immutable object
        y : immutable object

        Returns
        -------
        None

        """
        # Initialize if they are not already in the collection
        for elt in [x, y]:
            if elt not in self:
                self.add(elt)

        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return
        if self._siz[xroot] < self._siz[yroot]:
            self._par[xroot] = yroot
            self._siz[yroot] += self._siz[xroot]
        else:
            self._par[yroot] = xroot
            self._siz[xroot] += self._siz[yroot]
        self.n_comps -= 1

    def component(self, x:T) -> ordered_set[T]:
        """Find the connected component containing the given element.

        Parameters
        ----------
        x : immutable object

        Returns
        -------
        Insertion ordered set, first element is the root.

        Raises
        ------
        ValueError
            If the given element is not found.

        """
        if x not in self:
            raise ValueError('{} is not an element'.format(x))
        root = self.find(x)
        r = ordered_set([self[root]])
        r.update(elt for elt in self._elts if self.find(elt) == root)
        return r

    def components(self) -> list[ordered_set[T]]:
        """Return the list of connected components.

        Returns
        -------
        list
            A list of insertion ordered set, first element is the root.
        """
        components_dict:dict[int, ordered_set[T]] = {}
        for elt in self._elts:
            root = self.find(elt)
            try:
                components: ordered_set[T] = components_dict[root]
            except KeyError:
                components = components_dict[root] = ordered_set([self[root]])
            components.add(elt)
        return list(components_dict.values())
    
    def copy(self) -> UnionFind[T]:
        new = UnionFind()
        new.n_elts = self.n_elts
        new.n_comps = self.n_comps
        new._next = self._next
        new._elts = self._elts.copy()
        new._indx = self._indx.copy()
        new._par = self._par.copy()
        new._siz = self._siz.copy()
        return new
    
    def _is_leaf(self, i:int) -> bool:
        try:
            self._par.index(i)
        except ValueError:
            return True
        return False

    # def _find_leaf(self, x:T) -> T:
    #     # this implementation has a poor algorithmic complexity :/
    #     # if the trees are hight, it can become catastrophic.
    #     nodes: ordered_set[int] = ordered_set([self._indx[x]])
    #     while nodes:
    #         index = nodes[0]
    #         del nodes.values[index]

    #         if self._is_leaf(index):
    #             return index
    #         else:
    #             nodes.update(i for i,v in enumerate(self._par) if v == index)
    #     assert False
    
    # def _switch(self, a:T, b:T) -> None:
    #     indexa = self._indx[a]
    #     indexb = self._indx[b]

    #     self._elts[indexa] = b
    #     self._elts[indexb] = a
    #     self._indx[a] = indexb
    #     self._indx[b] = indexa

    
    # def remove(self, x: T) -> None:
    #     # http://www.corelab.ntua.gr/acac10/ACAC2010_Talks/SimonYoffe.pdf
    #     # https://citeseerx.ist.psu.edu/doc/10.1.1.13.4443

    #     if x not in self:
    #         raise ValueError('{} is not an element'.format(x))
    #     leaf = self._find_leaf(x)
    #     if leaf is not x:
    #         self._switch(leaf, x)
        
    #     index = self._indx[x]
    #     self._elts[index] = None
    #     #   switch leaf and x, x is now a leaf
    #     # place None at the index where x is stored.

    # def replace(self, a:T, b:T) -> None:
    #     ...
    #     # if a not in self:
    #     #     raise ValueError('{} is not an element'.format(a))
    #     # if b in self:
    #     #     raise ValueError('{} is already an element'.format(b))
    #     # index = self._indx[a]
    #     # self._elts[index] = b
    #     # self._indx[b] = index
    #     # del self._indx[a]