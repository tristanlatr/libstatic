from setuptools import setup
setup( 
    name = "libstatic",
    version = "0.1",
    description = "Lightweight static analysis library.",
    author = "Various Developers",
    url = "https://github.com/tristanlatr/libstatic",
    packages = ['libstatic'],
    install_requires = [
        'attrs',
        'astor',
        'typeshed_client',
        'beniget',
        'typing_extensions',
        'ast2json',
        'json2ast',
        'diskcache',
        'appdirs', 
        ],
    )