
import sys
import ast
from textwrap import dedent
from unittest import TestCase
import pytest


class TestScopeTree(TestCase):
    @pytest.mark.skipif(sys.version_info < (3, 10), reason='uses match-case syntax')
    def test_scopes(self, ):
        from libstatic._lib.scopetree import GlobalScope, ClassScope, FunctionScope

        # Set up a sample program
        # class C:
        #   def foo(self, a = blah):
        #     global x
        #     x = a

        globals = GlobalScope()
        c = ClassScope("C", globals)
        foo = FunctionScope("foo", c)
        foo.store("self")
        foo.store("a")
        foo.add_global("x")

        assert foo.lookup("C") is globals
        assert c.lookup("C") is globals

        assert foo.lookup("foo") is None
        assert c.lookup("foo") is c

        assert foo.lookup("self") is foo
        assert c.lookup("self") is None

        assert foo.lookup("a") is foo
        assert c.lookup("a") is None

        assert foo.lookup("blah") is None
        assert c.lookup("blah") is None

        assert foo.lookup("x") is globals
        assert c.lookup("x") is None
    
    @pytest.mark.skipif(sys.version_info < (3, 10), reason='uses match-case syntax')
    def test_builder(self, ):
        from libstatic._lib.scopetree import Builder

        src = '''
        class C:
          def foo(self, a = blah):
            global x
            x = a
        '''
        node = ast.parse(dedent(src))
        builder = Builder()
        builder.build(node)

        globals = builder.globals
        clsnode = node.body[0]
        c = builder.scopes[clsnode]
        foo = builder.scopes[clsnode.body[0]]
        assert c and foo
       
        assert foo.lookup("C") is globals
        assert c.lookup("C") is globals

        assert foo.lookup("foo") is None
        assert c.lookup("foo") is c

        assert foo.lookup("self") is foo
        assert sorted(globals.locals) == ['C', ]
        assert sorted(c.locals) == ['foo', ]
        assert c.lookup("self") is None

        assert foo.lookup("a") is foo
        assert c.lookup("a") is None

        assert foo.lookup("blah") is None
        assert c.lookup("blah") is None

        assert foo.lookup("x") is globals
        assert c.lookup("x") is None