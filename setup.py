from setuptools import setup
setup( 
    name = "libstatic",
    version = "0.2.0.dev.0",
    description = "Static analysis library based on beginet",
    author = "Various Developers",
    url = "https://github.com/tristanlatr/libstatic",
    packages = ['libstatic'],
    install_requires = [
        'attrs',
        'typeshed_client',
        'beniget',
        'astor',
        ],
    python_requires='>=3.7',
    )