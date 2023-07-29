from typing import Optional
import ast

from .exceptions import StaticCodeUnsupported


# This could be implemented with the use-def chains.
# The def of a lhs value of an assignment (store) is the rhs,
# but this implementation might do the job just as well.
def get_stored_value(
    node: ast.Name, assign: "ast.Assign|ast.AnnAssign"
) -> Optional[ast.expr]:
    """
    Given an ast.Name instance with Store context and it's assignment statement,
    figure out the right hand side expression that is stored in the symbol.

    Limitation:
        - Since this function is mainly used to evaluate constant values,
          starred assignments are not supported as they ususally mean
          we're unpacking something of variable lenght.
        - Nested tuple assignments are not supported.
        - For loops targets are not supported by this function,
          it need to return an object that represent an expression of type T,
          not the expression itself, since it usually have multiple values.

    :raises StaticCodeUnsupported: If there is no obvious value
    """

    # There is no augmented assignments
    value = assign.value
    for target in assign.targets if isinstance(assign, ast.Assign) else [assign.target]:
        if target is node:
            return value
        elif (
            isinstance(target, (ast.List, ast.Tuple))
            and isinstance(value, (ast.List, ast.Tuple))
            and len(target.elts) == len(value.elts)
        ):
            try:
                # only first level lists are checked.
                index = target.elts.index(node)
            except ValueError:
                continue
            else:
                # can't raise IndexError since we checked len(target.elts)==len(value.elts)
                element = value.elts[index]
                if isinstance(element, ast.Starred):
                    continue
                return element
    raise StaticCodeUnsupported(assign, "assignment")
