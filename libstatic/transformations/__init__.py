from __future__ import annotations

from .. import passmanager

# from libstatic._lib.imports import ParseImportedNames, ImportInfo
# from libstatic._lib.ivars import _compute_ivars
# from libstatic._lib import exceptions

################## Transformations

from libstatic._lib.transform import Transform
@passmanager.transformation(on=passmanager.Tree)
def normalize(_, node: passmanager.Tree):
    transformer = Transform()
    transformer.transform(node.root)
    yield 'update', transformer.update