from setuptools import setup, find_packages
setup( 
    name = "libstatic",
    version = "0.2.0.dev4",
    description = "Static analysis framework",
    maintainer = "tristanlatr",
    url = "https://github.com/tristanlatr/libstatic",
    packages = find_packages(exclude=['tests']),
    install_requires = [
        'attrs',
        'typeshed_client',
        # 'beniget @ git+https://github.com/serge-sans-paille/beniget.git',
        'beniget @ git+https://github.com/pyforks/beniget-ng.git',
        'astor',
        ],
    python_requires='>=3.7',
    )