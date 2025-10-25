from __future__ import annotations
from enum import Enum, auto
from typing import Sequence

from attrs import define
from docutils.nodes import document

@define
class Serializable:
    ...

@define
class Edge(Serializable):
    ...
    # there are three types if edges: 
    # - edges that link one vertex to a value directly with to_value
    # - edges that link one vertex to a single vertex with to_id
    # - edges that link one vertex to multiple vertexes with to_ids

@define
class Vertex(Serializable):
    id: object

# Symbol

@define
class Symbol(Vertex):
    name:str
    kind: Kind
    semantics: list[Semantics]
    visibility: Visibility

# Symbol Kind

class Kind(Enum):
    MODULE = auto()
    CLASS = auto()
    FUNCTION = auto()
    PARAMETER = auto()
    ATTRIBUTE = auto()
    INDIRECTION = auto()

# Symbol Semantics

class Semantics(Enum):
    TYPE_CHECK_ONLY = auto()
    WILDCARD_EXPOSED = auto()
    FINAL = auto()
    ABSTRACT = auto()
    EXTERNAL = auto()

    # Module semantics
    PACKAGE = auto()
    NAMESPACE_PACKAGE = auto()

    # Class semantics
    INTERFACE = auto()
    EXCEPTION = auto()

    ENUM = auto()
    DATACLASS = auto()

    # Function semantics
    METHOD = auto()
    CLASS_METHOD = auto()
    STATIC_METHOD = auto()
    GETTER = auto()
    SETTER = auto()
    DELETER = auto()
    COROUTINE = auto()
    NO_RETURN = auto()

    # Parameter semantics
    SELF = auto()
    CLS = auto()

    # Attribute semantics
    PROPERTY = auto()
    INSTANCE_ATTRIBUTE = auto()
    CLASS_ATTRIBUTE = auto()
    CONSTANT = auto()
    DESCRIPTOR = auto()
    TYPE_ALIAS = auto()

    TYPE_VAR = auto()
    TYPE_VAR_TUPLE = auto()
    PARAM_SPEC = auto()

    # Attribute / Indirection semantics
    ALIAS = auto()

    # Indirection semantics
    IMPORT = auto()
    INHERITED_MEMBER = auto()

# Symbol visibility

class Visibility(Enum):
    HIDDEN = auto()
    PRIVATE = auto()
    PUBLIC = auto()

# Symbol modifiers

class Modifiers(Enum):
    ASYNC = auto()
    TYPE = auto()

# Parameter kind

class ParameterKind(Enum):
    POSITIONAL_ONLY = auto()
    POSITIONAL = auto()
    POSITIONAL_REMAINDER = auto()
    KEYWORD_ONLY = auto()
    KEYWORD_REMAINDER = auto()

@define
class ModifiersOf(Edge):
    from_id: object #: from attribute or function Symbol
    to_value: Sequence[Modifiers]

# Locations

@define
class Offset(Vertex):
    linenumber: int
    column: int

@define
class Location(Vertex):
    path: str
    source_href: str

@define
class NsPackageLocations(Vertex):
    paths: list[str]
    source_hrefs: list[str]
    # special of case of namespace packages prescribes the use of several 
    # paths and hrefs

@define
class StartOf(Edge):
    from_id: object #: from Location
    to_id: object #: to Offset

@define
class EndOf(Edge):
    from_id: object #: from Location
    to_id: object #: to Offset

@define
class LocationOf(Edge): # Applicable only for Symbol and Docstrings
    from_id: object # from Vertex
    to_id: object # to Location or NamespacePackageLocation

# Docstrings

@define
class Docstring(Vertex):
    contents: str

@define
class DocstringOf(Edge):
    from_id: object #: from Symbol
    to_id: object #: to Docstring

# Children / Parents

@define
class ChildrenOf(Edge):
    from_id: object
    to_ids: Sequence[object]

@define # reverse of ChildrenOf
class ParentOf(Edge):
    from_id: object
    to_id: object

# Renderable bits, this one is a little bit special 
# because it encapsulates the element inside a document instance that is suitable
# for converting to HTML, text and others and IIF the document represents
# an AST expressions, back to AST with something like:: 
#   ast.parse(to_text(renderable)).body[0].value
# We use docutils's node title_reference to accomodate use names and store the definitions
# ID(s) in the special "def_ids" attribute, which is a sequence of IDs of Symbols. 
# So this vertex type has it's edges encapsulated directly in the vertex itseft. 

@define
class Renderable(Vertex):
    document: document

@define
class ParsedDocstringOf(Edge):
    from_id: object #: from Docstring
    to_id: object #: to Renderable


# Decorators

@define
class DecoratorsOf(Edge):
    from_id: object # from Symbol
    to_ids: Sequence[object] # to Renderables of AST

# Deprecations

@define
class Deprecation(Vertex):
    since: str
    replacement: str

@define
class DeprecationOf(Edge):
    from_id: object #: from Symbol
    to_id: object #: to Deprecation

# Docsources

@define
class DocsourcesOf(Edge):
    from_id: object #: from Symbol
    to_ids: Sequence[object] #: to Symbols

# Overrides

@define
class Overrides(Edge):
    from_id: object #: from Symbol
    to_id: object #: to Symbol

@define # reverse of Overrides
class OverridenBy(Edge):
    from_id: object #: from Symbol
    to_ids: Sequence[object] #: to Symbols

# Implements

@define
class Implements(Edge):
    from_id: object #: from Symbol
    to_id: object #: to Symbol

@define # reverse of Implements
class ImplementedBy(Edge):
    from_id: object #: from Symbol
    to_ids: Sequence[object] #: to Symbols

# Submodules

@define
class SubmodulesOf(Edge):
    from_id: object #: from module Symbol
    to_ids: Sequence[object] #: to module Symbols

# Bases

@define
class RawBasesOf(Edge):
    from_id: object #: from class Symbol
    to_ids: Sequence[object] #: to Renderables

@define
class BasesOf(Edge):
    from_id: object #: from class Symbol
    to_ids: Sequence[object] #: to Symbols

@define # reverse of BasesOf
class SubclassesOf(Edge):
    from_id: object #: from class Symbol
    to_ids: Sequence[object] #: to class Symbols

@define
class MroOf(Edge):
    from_id: object #: from class Symbol
    to_ids: Sequence[object] #: to class Symbols

# Other class edges

@define
class ConstructorsOf(Edge):
    from_id: object #: from class Symbol
    to_ids: Sequence[object] #: to function Symbols

@define
class RawMetaclassOf(Edge):
    from_id: object #: from class Symbol
    to_id: object #: to Renderable

@define
class MetaclassOf(Edge):
    from_id: object #: from class Symbol
    to_id: object #: to Symbol

# PEP695 type parameters, applicable to attribute/function/class edges

@define
class RawTypeParametersOf(Edge):
    from_id: object #: from classes or attributes Symbols
    to_ids: Sequence[object] #: to Renderables

@define
class TypeParametersOf(Edge):
    from_id: object #: from attribute/function/class Symbol
    to_ids: Sequence[object] #: to type var/type var tuple/param spec Symbols

# Function edges

@define
class OverloadsOf(Edge): 
    from_id: object #: from function Symbol
    to_ids: Sequence[object] #: to function Symbols

@define # reverse of OverloadsOf
class PrimaryOf(Edge): 
    from_id: object #: from function Symbol
    to_id: object #: to function Symbol

@define
class ParametersOf(Edge): 
    from_id: object #: from function Symbol
    to_ids: Sequence[object] #: to parameter Symbols

@define
class ReturnTypeOf(Edge): 
    from_id: object #: from function Symbol
    to_id: object #: to Renderable

# Type variables edges

@define
class BoundOf(Edge):
    from_id: object #: from type parameter Symbol
    to_id: object #: to Renderable

# Type variables / parameter edges

@define
class DefaultValueOf(Edge):
    from_id: object #: from (type)parameters Symbol
    to_id: object #: to Renderable

# Parameter edges

@define
class ParameterKindOf(Edge):
    from_id: object #: from parameter Symbol
    to_value: ParameterKind # to kind

# Parameter / Attribute edges

@define
class AnnotationOf(Edge):
    from_id: object #: from parameter/attribute Symbol
    to_id: object #: to Renderable

# Attribute edges

@define
class ValueOf(Edge):
    from_id: object #: from attribute Symbol
    to_id: object #: to Renderable

# Indirection / Attribute alias edges

@define
class DefinitionsOf(Edge): 
    from_id: object # from indirection or alias Symbol
    to_ids: Sequence[object] #: to Symbols

# Indirection import edges 

@define
class OrgModuleOf(Edge): 
    from_id: object #: from import Symbol
    to_value: str #: to string

@define
class OrgNameOf(Edge): 
    from_id: object #: from import Symbol
    to_value: str #: to string

@define
class AsNameOf(Edge): 
    from_id: object #: from import Symbol
    to_value: str #: to string

# Indirection inherited edges 

@define
class InheritedFrom(Edge): 
    from_id: object #: from Symbol
    to_id: object #: to class Symbol

# Re-export informations

@define
class ReExportedTo(Edge):
    from_id: object #: from attribute/function/class/module Symbol
    to_ids: Sequence[object] #: to indirection Symbols

