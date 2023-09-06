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
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=10>
>>> uf.union('e', 'd')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 2, 1, 1, 1, 1, 1],
par=[0, 1, 2, 4, 4, 5, 6, 7, 8, 9],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=9>
>>> uf.union('d', 'i')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 3, 1, 1, 1, 1, 1],
par=[0, 1, 2, 4, 4, 5, 6, 7, 4, 9],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=8>
>>> uf.union('g', 'f')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 3, 1, 2, 1, 1, 1],
par=[0, 1, 2, 4, 4, 6, 6, 7, 4, 9],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=7>
>>> uf.union('j', 'e')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 1, 1, 4, 1, 2, 1, 1, 1],
par=[0, 1, 2, 4, 4, 6, 6, 7, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=6>
>>> uf.union('c', 'b')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 2, 1, 4, 1, 2, 1, 1, 1],
par=[0, 2, 2, 4, 4, 6, 6, 7, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=5>
>>> uf.union('i', 'j')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 2, 1, 4, 1, 2, 1, 1, 1],
par=[0, 2, 2, 4, 4, 6, 6, 7, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=5>
>>> uf.union('f', 'a')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 2, 1, 4, 1, 3, 1, 1, 1],
par=[6, 2, 2, 4, 4, 6, 6, 7, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=4>
>>> uf.union('h', 'c')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 3, 1, 1, 1],
par=[6, 2, 2, 4, 4, 6, 6, 2, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=3>
>>> uf.union('g', 'b')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 2, 6, 4, 4, 6, 6, 2, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=2>
>>> uf.union('a', 'b')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 6, 6, 4, 4, 6, 6, 2, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=2>
>>> uf.union('g', 'h')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 6, 6, 4, 4, 6, 6, 6, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[],
n_elts=10,n_comps=2>
>>> uf.connected('a', 'g')
True
>>> uf.component('a')
<ordered_set {g, a, b, c, f, h}>
>>> uf.components()
[<ordered_set {g, a, b, c, f, h}>, <ordered_set {e, d, i, j}>]
>>> uf.remove('i')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 6, 6, 4, 4, 6, 6, 6, 4, 4],
nb_rm=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
removed=[8],
free=[],
n_elts=9,n_comps=2>
>>> uf.components()
[<ordered_set {g, a, b, c, f, h}>, <ordered_set {e, d, j}>]
>>> uf.remove('j')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 1],
par=[6, 6, 6, 4, 4, 6, 6, 6, 4, 4],
nb_rm=[0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
removed=[8, 9],
free=[],
n_elts=8,n_comps=2>
>>> uf.components()
[<ordered_set {g, a, b, c, f, h}>, <ordered_set {e, d}>]
>>> uf.remove('d')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', None, 'e', 'f', 'g', 'h', None, None],
siz=[1, 1, 3, 0, 2, 1, 6, 1, 0, 0],
par=[6, 6, 6, 3, 4, 6, 6, 6, 8, 9],
nb_rm=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
removed=[],
free=[3, 8, 9],
n_elts=7,n_comps=2>
>>> uf.components()
[<ordered_set {g, a, b, c, f, h}>, <ordered_set {e}>]
>>> uf.component('e')
<ordered_set {e}>
>>> uf.union('e', 'z')
>>> uf.remove('a')
>>> uf.union('z', 'x')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'z', 'e', 'f', 'g', 'h', 'x', None],
siz=[1, 1, 3, 1, 4, 1, 6, 1, 1, 0],
par=[6, 6, 6, 4, 4, 6, 6, 6, 4, 9],
nb_rm=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
removed=[0],
free=[9],
n_elts=8,n_comps=2>
>>> uf.components()
[<ordered_set {g, b, c, f, h}>, <ordered_set {e, z, x}>]
>>> uf.remove('g')
>>> uf.components()
[<ordered_set {b, c, f, h}>, <ordered_set {e, z, x}>]
>>> uf.union('x', 'y')
>>> uf
<UnionFind:
elts=['a', 'b', 'c', 'z', 'e', 'f', 'g', 'h', 'x', 'y'],
siz=[1, 1, 3, 1, 5, 1, 6, 1, 1, 1],
par=[6, 6, 6, 4, 4, 6, 6, 6, 4, 4],
nb_rm=[0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
removed=[0, 6],
free=[],
n_elts=8,n_comps=2>
>>> for e in 'bcfhe':
...     uf.remove(e)
...
>>> uf
<UnionFind:
elts=[None, None, None, 'z', 'e', 'f', None, 'h', 'x', 'y'],
siz=[0, 0, 0, 1, 5, 3, 0, 1, 1, 1],
par=[0, 1, 2, 4, 4, 5, 6, 5, 4, 4],
nb_rm=[0, 0, 0, 0, 1, 2, 0, 0, 0, 0],
removed=[5, 7, 4],
free=[6, 0, 1, 2],
n_elts=3,n_comps=2>
>>> uf.components()
[<ordered_set {z, x, y}>]

"""
from __future__ import annotations

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
        
        # support for remove()
        # based on http://www.corelab.ntua.gr/acac10/ACAC2010_Talks/SimonYoffe.pdf
        # section 4. Union-Find via path compression and linking by rank size
        self._nb_rm: list[int] = [] # number of elements removed - correct only for roots
        self._removed: set[int] = ordered_set() # indexes of removed items
        self._free: set[int] = ordered_set() # indexes of freed spot in the element list after rebuild

        if elements is None:
            elements = []
        for elt in elements:
            self.add(elt)


    def __repr__(self) -> str:
        return  (
            '<UnionFind:\nelts={},\nsiz={},\npar={},\nnb_rm={},\nremoved={},\nfree={},\nn_elts={},n_comps={}>'
            .format(
                self._elts,
                self._siz,
                self._par,
                list(self._nb_rm),
                list(self._removed),
                list(self._free),
                self.n_elts,
                self.n_comps,
            ))

    def __len__(self) -> int:
        return self.n_elts

    def __contains__(self, x) -> bool:
        in_index = x in self._indx
        return in_index and self._indx[x] not in self._removed

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
        
        if self._free:
            index = next(iter(self._free))
            self._free.discard(index)
            self._elts[index] = x
            self._indx[x] = index
            self._par[index] = index
            self._siz[index] = 1
            self._nb_rm[index] = 0
            self.n_elts += 1
            self.n_comps += 1

        else:
            self._elts.append(x)
            self._indx[x] = self._next
            self._par.append(self._next)
            self._siz.append(1)
            self._nb_rm.append(0)
            self._next += 1
            self.n_elts += 1
            self.n_comps += 1

    def _find(self, x: T) -> int:
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
        return self._find(x) == self._find(y)

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

        xroot = self._find(x)
        yroot = self._find(y)
        if xroot == yroot:
            return
        if self._siz[xroot] < self._siz[yroot]:
            self._par[xroot] = yroot
            self._siz[yroot] += self._siz[xroot]
            self._nb_rm[yroot] += self._nb_rm[xroot]
        else:
            self._par[yroot] = xroot
            self._siz[xroot] += self._siz[yroot]
            self._nb_rm[xroot] += self._nb_rm[yroot]
        self.n_comps -= 1

    def component(self, x:T, _include_rm:bool=False) -> ordered_set[T]:
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
        if x not in (self._elts if _include_rm else self):
            raise ValueError('{} is not an element'.format(x))
        
        root = self._find(x)
        if not _include_rm and root in self._removed:
            s = ordered_set()
        else:
            s = ordered_set(self[root])
        s.update(elt for i, elt in enumerate(self)
                 if elt is not None and ((not _include_rm or i not in self._removed) or _include_rm)
                     and self._find(elt) == root)
        return s

    def components(self) -> list[ordered_set[T]]:
        """Return the list of connected components.

        Returns
        -------
        list
            A list of insertion ordered set, first element is the root.
        """
        components_dict:dict[int, ordered_set[T]] = {}
        for i, elt in enumerate(self._elts):
            if elt is None:
                continue
            if i in self._removed:
                continue
            root = self._find(elt)
            try:
                components: ordered_set[T] = components_dict[root]
            except KeyError:
                if root not in self._removed:
                    components = components_dict[root] = ordered_set([self[root]])
                else:
                    components = components_dict[root] = ordered_set()
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
        new._nb_rm = self._nb_rm.copy()
        new._removed = self._removed.copy()
        new._free = self._free.copy()
        return new
    
    def remove(self, x: T) -> None:
        # http://www.corelab.ntua.gr/acac10/ACAC2010_Talks/SimonYoffe.pdf
        # section 4. Union-Find via path compression and linking by rank size
        
        if x not in self:
            raise ValueError('{} is not an element'.format(x))
        
        index = self._indx[x]
        root = self._find(x)
        self._removed.add(index)
        self._nb_rm[root] += 1
        self.n_elts -= 1
        
        siz_by_2 = round(self._siz[root] / 2)
        if self._nb_rm[root] > siz_by_2:
            # rebuild
            rootindex = None
            for i in self.component(self[root], _include_rm=True):
                iindx = self._indx[i]
                
                if iindx in self._removed:
                    self._elts[iindx] = None
                    self._par[iindx] = iindx
                    self._siz[iindx] = 0
                    self._nb_rm[iindx] = 0
                    del self._indx[i]

                    self._free.add(iindx)
                    self._removed.discard(iindx)
                    continue
                
                if rootindex is None:
                    self._siz[iindx] = siz_by_2
                    self._nb_rm[iindx] = 0
                    self._par[iindx] = iindx
                    rootindex = iindx
                else:
                    self._par[iindx] = rootindex
