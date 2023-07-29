from setuptools import setup
setup( 
    name = "libstatic",
    version = "0.1",
    description = "Static analysis library based on beginet",
    author = "Various Developers",
    url = "https://github.com/tristanlatr/libstatic",
    packages = ['libstatic'],
    install_requires = [
        'attrs',
        'typeshed_client',
        'beniget',
        'typing_extensions',
        'astor',
        ],
    )