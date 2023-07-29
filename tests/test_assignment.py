import ast
from unittest import TestCase

from libstatic.assignment import get_stored_value
from libstatic.ancestors import Ancestors
from libstatic.exceptions import StaticCodeUnsupported

class TestGetStoredValue(TestCase):
    def test_simple_assignment(self):
        code = 'a = 2'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)
        
        name = next((n for n in ast.walk(node) if isinstance(n, ast.Name)))
        constant = next((n for n in ast.walk(node) if isinstance(n, (ast.Constant, ast.Num))))
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))
        
        assert get_stored_value(name, assignment) is constant
        assert get_stored_value(assignment.targets[0], assignment) is constant
    
    def test_no_value_assignment(self):
        code = 'a:int'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)
        
        name = next((n for n in ast.walk(node) if isinstance(n, ast.Name)))
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.AnnAssign)))
        
        assert get_stored_value(name, assignment) is None
        assert get_stored_value(assignment.target, assignment) is None
    
    def test_tuple_assignment(self):
        code = 'a,b = 2,3'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)
        
        name = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        constant = [n for n in ast.walk(node) if isinstance(n, (ast.Constant, ast.Num))][0]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        assert get_stored_value(name, assignment) is constant

        name = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        constant = [n for n in ast.walk(node) if isinstance(n, (ast.Constant, ast.Num))][1]
        assert get_stored_value(name, assignment) is constant

        tuple_ltarget = [n for n in ast.walk(node) if isinstance(n, ast.Tuple)][0]
        tuple_rvalue = [n for n in ast.walk(node) if isinstance(n, ast.Tuple)][1]
        assert get_stored_value(tuple_ltarget, assignment) is tuple_rvalue
    
    def test_unsupported_assignment_star_value(self):
        code = 'd,e=(*a,)'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        d = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        e = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        with self.assertRaises(StaticCodeUnsupported, msg='unsupported assignment'):
            get_stored_value(d, assignment)
        
        with self.assertRaises(StaticCodeUnsupported, msg='unsupported assignment'):
            get_stored_value(e, assignment)

    def test_unsupported_nested_assignment(self):
        code = 'd,e,(f,g)=(1,2,(3,4))'

        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        d = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        e = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        assert get_stored_value(d, assignment).__class__.__name__ in ('Constant', 'Num')
        assert get_stored_value(e, assignment).__class__.__name__ in ('Constant', 'Num')

        f = [n for n in ast.walk(node) if isinstance(n, ast.Name)][2]
        g = [n for n in ast.walk(node) if isinstance(n, ast.Name)][3]

        with self.assertRaises(StaticCodeUnsupported, msg='unsupported assignment'):
            get_stored_value(f, assignment)
        
        with self.assertRaises(StaticCodeUnsupported, msg='unsupported assignment'):
            get_stored_value(g, assignment)

    def test_unsupported_assignment_unpack(self):
        code = 'a,b=c'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        a = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        with self.assertRaises(StaticCodeUnsupported, msg='unsupported assignment'):
            get_stored_value(a, assignment)

        code = '*a,c,b=(1,2,3,4,5,6)'
        node = ast.parse(code)
        ancestors = Ancestors()
        ancestors.visit(node)

        a = [n for n in ast.walk(node) if isinstance(n, ast.Name)][0]
        c = [n for n in ast.walk(node) if isinstance(n, ast.Name)][1]
        assignment = next((n for n in ast.walk(node) if isinstance(n, ast.Assign)))

        with self.assertRaises(StaticCodeUnsupported, msg='unsupported assignment'):
            get_stored_value(a, assignment)
        with self.assertRaises(StaticCodeUnsupported, msg='unsupported assignment'):
            get_stored_value(c, assignment)
