from setuptools import setup, find_packages
setup( 
    name = "libstatic",
    version = "0.2.0.dev2",
    description = "Static analysis library based on beginet",
    author = "Various Developers",
    url = "https://github.com/tristanlatr/libstatic",
    packages = find_packages(exclude=['tests']),
    install_requires = [
        'attrs',
        'typeshed_client',
        'beniget',
        'astor',
        ],
    python_requires='>=3.7',
    )