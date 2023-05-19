import gast as ast
from unittest import TestCase
from textwrap import dedent, indent


from libstatic.transform import Transform


class TestPrepare(TestCase):
    def checkTransforms(
        self, code, ref, module=True, function=True, klass=True, method=True
    ):
        code = dedent(code)
        ref = dedent(ref)

        def check(code, ref, expect_success):
            node = ast.parse(code)
            Transform().transform(node)
            if expect_success:
                self.assertEqual(ast.unparse(node).strip(), ref.strip())
            else:
                self.assertNotEqual(ast.unparse(node).strip(), ref.strip())

        check(code, ref, expect_success=module)

        check(
            "def f():" + indent(code, prefix="    "),
            "def f():" + indent(ref, prefix="    "),
            expect_success=function,
        )

        check(
            "class S:" + indent(code, prefix="    "),
            "class S:" + indent(ref, prefix="    "),
            expect_success=klass,
        )

        check(
            "class S:\n\n    def f(self):" + indent(code, prefix="        "),
            "class S:\n\n    def f(self):" + indent(ref, prefix="        "),
            expect_success=method,
        )

    def test_IfBanchDead(self):
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

    def test_BothIfElseBranchesDead(self):
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

    def test_IfElseInvalidJump(self):
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

    def test_WhileIfJump(self):
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

    def test_DeadYield(self):
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

    def test_AugAssign(self):
        code = """
        a = 1
        a += 1
        """

        transformed = """
        a = 1
        a = a + 1
        """
        self.checkTransforms(code, transformed)

    def test_DunderAllTransform(self):
        code = """
        __all__ = []
        __all__.append('2')
        __all__.extend(['2'])
        """

        transformed = """
        __all__ = []
        __all__ = __all__ + ['2']
        __all__ = __all__ + ['2']
        """
        self.checkTransforms(code, transformed, function=False, klass=False, method=False)
    
    def test_ReguluarList(self):
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
