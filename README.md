# libstatic
Lightweight static analysis library.

The purpose of this module is to provide a simple and easy to use library that help with understanding python code. Especially helps to answer questions like: 
- Is this variable has only one possible value? 
- What is the value of `__all__` variable?
- Is a name defined?

- What is the type of an unnatotated variable?
- Has a name been deleted?
- What constraints are applicable to this statement?

## Core principles
- Python is highly dymanmic language, so this library WILL make mistakes (for instance when a module level list is modified inside a function), but it will get the right values in 90% of the cases, which is more than sufficient
