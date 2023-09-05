from libstatic._analyzer.unionfind import UnionFind

def test_unionfind():
    u = UnionFind()
    assert u.n_comps == 0
    assert len(u) == 0
    u.union(1, 2)
    assert u.n_comps == 1
    assert len(u) == 2
    u.union(2, 3)
    assert u.n_comps == 1
    assert len(u) == 3

    u.union(4, 5)
    assert u.n_comps == 2
    assert len(u) == 5

    u.union(6, 7)
    assert u.n_comps == 3
    assert len(u) == 7
    u.union(7, 8)
    assert u.n_comps == 3
    assert len(u) == 8
    u.union(8, 9)
    assert u.n_comps == 3
    assert len(u) == 9

    u.union(8, 9)  # duplicate
    assert u.n_comps == 3
    assert len(u) == 9

    u.add(10)
    assert u.n_comps == 4
    assert len(u) == 10

    u.union(4, 7)
    assert u.n_comps == 3
    assert len(u) == 10