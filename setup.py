from setuptools import setup, find_packages
setup( 
    name = "libstatic",
    version = "0.2.0.dev3",
    description = "Static analysis framework",
    author = "Various Developers",
    url = "https://github.com/tristanlatr/libstatic",
    packages = find_packages(exclude=['tests']),
    install_requires = [
        'attrs',
        'typeshed_client',
        # TODO: Finish the work on beniget and use the upstream version 
        # 'beniget @ git+https://github.com/serge-sans-paille/beniget.git',
        'beniget @ git+https://github.com/pyforks/beniget-ng.git',
        'astor', # TODO: Drop python 3.8 and remove the astor dependency.
        ],
    python_requires='>=3.7',
    )