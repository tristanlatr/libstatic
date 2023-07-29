import ast
from unittest import TestCase
from textwrap import dedent, indent


from libstatic.transform import Transform
from libstatic.shared import unparse


class TestPrepare(TestCase):
    def checkTransforms(
        self, code, ref, module=True, function=True, klass=True, method=True
    ) -> None:
        code = dedent(code)
        ref = dedent(ref)

        def check(code, ref, expect_success) -> None:
            node = ast.parse(code)
            Transform().transform(node)
            unparsed = '\n'.join(line for line in unparse(node).strip().splitlines() if line)
            ref = '\n'.join(line for line in ref.strip().splitlines() if line)
            if expect_success:
                self.assertEqual(unparsed, ref)
            else:
                self.assertNotEqual(unparsed, ref)

        check(code, ref, expect_success=module)

        check(
            "def f():\n" + indent(code, prefix="    "),
            "def f():\n" + indent(ref, prefix="    "),
            expect_success=function,
        )

        check(
            "class S:\n" + indent(code, prefix="    "),
            "class S:\n" + indent(ref, prefix="    "),
            expect_success=klass,
        )

        check(
            "class S:\n\n    def f(self):\n" + indent(code, prefix="        "),
            "class S:\n\n    def f(self):\n" + indent(ref, prefix="        "),
            expect_success=method,
        )

    def test_IfBanchDead(self) -> None:
        # dead code is removed when it's directly after a control flow jump.
        code = """
        if sys.version_info.major == 2:
            raise RuntimeError()
            dead = 'code'
        """

        transformed = """
        if sys.version_info.major == 2:
            raise RuntimeError()
        """
        self.checkTransforms(code, transformed)
        self.checkTransforms(
            code.replace("raise RuntimeError()", "assert False"), 
            transformed.replace("raise RuntimeError()", "assert False")
        )

    def test_BothIfElseBranchesDead(self) -> None:
        # dead code is removed when both branches of a if/else branch are dead.
        code = """
        if sys.version_info.major == 2:
            raise RuntimeError()
        else:
            raise TypeError()
        dead = 'code'
        """

        transformed = """
        if sys.version_info.major == 2:
            raise RuntimeError()
        else:
            raise TypeError()
        """
        self.checkTransforms(code, transformed)
        self.checkTransforms(
            code.replace("raise RuntimeError()", "assert False"), 
            transformed.replace("raise RuntimeError()", "assert False")
        )

    def test_IfElseInvalidJump(self) -> None:
        # bogus control flow jumps are not considered.
        code = """
        if sys.version_info.major == 2:
            raise RuntimeError()
        else:
            break
        boggus = 'code'
        """

        transformed = """
        if sys.version_info.major == 2:
            raise RuntimeError()
        else:
            break
        boggus = 'code'
        """
        self.checkTransforms(code, transformed)
        self.checkTransforms(
            code.replace("beak", "continue"), transformed.replace("beak", "continue")
        )
        self.checkTransforms(
            code.replace("raise RuntimeError()", "assert False"), 
            transformed.replace("raise RuntimeError()", "assert False")
        )

    def test_WhileIfJump(self) -> None:
        # nothing is removed when there is nothing to remove
        code = """
        while True:
            line = fp.readline()
            if not line:
                break
        """

        transformed = """
        while True:
            line = fp.readline()
            if not line:
                break
        """
        self.checkTransforms(code, transformed)
        self.checkTransforms(
            code.replace("beak", "continue"), transformed.replace("beak", "continue")
        )

    def test_DeadYield(self) -> None:
        # a dead yield is not removed since it's semantically important to make a 
        # function into a generator.
        code = """
        raise NotImplemented()
        yield
        """

        transformed = """
        raise NotImplemented()
        yield
        """
        self.checkTransforms(code, transformed)
        self.checkTransforms(
            code.replace("yield", "yield from ()"),
            transformed.replace("yield", "yield from ()"),
        )

    def test_AugAssign(self) -> None:
        # regular augmented assignments are not transformed.
        code = """
        a = 1
        a += 1
        """

        transformed = """
        a = 1
        a += 1
        """
        self.checkTransforms(code, transformed)
    
    def test_AugAssignDunderAll(self) -> None:
        # augmented assignments of module level variable __all__ are transformed into regular assignments.
        code = """
        __all__ = 1
        __all__ += 1
        """

        transformed = """
        __all__ = 1
        __all__ = __all__ + 1
        """
        self.checkTransforms(code, transformed,
                             function=False, klass=False, method=False)
    
    def test_AugAssignDunderAllInScope(self) -> None:
        # augmented assignments of variable __all__ inside classes or functions are not transformed.
        code = """
        def f():
            __all__ = 1
            __all__ += 1
        """

        transformed = """
        def f():
            __all__ = 1
            __all__ += 1
        """
        self.checkTransforms(code, transformed)

        code = """
        class C:
            __all__ = 1
            __all__ += 1
        """

        transformed = """
        class C:
            __all__ = 1
            __all__ += 1
        """
        self.checkTransforms(code, transformed)

    def test_DunderAllTransform(self) -> None:
        # list modifications of module level variable __all__ are transformed into regumar assignments.
        code = """
        __all__ = []
        __all__.append('2')
        __all__.extend(['2'])
        """

        transformed = """
        __all__ = []
        __all__ = [*__all__, '2']
        __all__ = [*__all__, *['2']]
        """
        self.checkTransforms(code, transformed, function=False, klass=False, method=False)
    
    def test_DunderAllTransformInScope(self) -> None:
        # list modifications of variable __all__ inside classes or functions are not transformed.
        code = """
        def f():
            __all__ = []
            __all__.append('2')
            __all__.extend(['2'])
        """

        transformed = """
        def f():
            __all__ = []
            __all__.append('2')
            __all__.extend(['2'])
        """

        self.checkTransforms(code, transformed)

        code = """
        class C:
            __all__ = []
            __all__.append('2')
            __all__.extend(['2'])
        """

        transformed = """
        class C:
            __all__ = []
            __all__.append('2')
            __all__.extend(['2'])
        """

        self.checkTransforms(code, transformed)
    
    def test_ReguluarList(self) -> None:
        # regular list modifications are not transformed.
        code = """
        a = []
        a.append('2')
        a.extend(['2'])
        """

        transformed = """
        a = []
        a.append('2')
        a.extend(['2'])
        """
        self.checkTransforms(code, transformed)