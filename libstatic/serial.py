"""
Serialization/De-serialization of L{State} objects
"""

# To serialize the chains, we need to build a serilizable model from it;
# because there are possiblity circular references in the model.
# each ast node should be given an id, then the whole model should use ids instead of ast references or def
# this means having our own version of ast2json that adds ids to all nodes, and also a custom
# json2ast that maps each id to the created node.
# this is should be composed by the filename/modname/node number
# also the def-use chains

# serialization must be done on a per-module basis
# eg pydoctor.model.json


class SerializableDef:
    ...


class SerializableMod:
    ...


class SerializableVar:
    ...


class SerializableImp:
    ...


class SerializableCls:
    ...


class SerializableFunc:
    ...


class SerializableAST:
    ...
