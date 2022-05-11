from setuptools import setup
from nirtools import __version__

install_requires = [
    "numpy>=1.19",
    "scipy>=1.5.4"
    "pandas==1.1.5",
    "tqdm>=4.54",
    "scikit-learn>=0.23.1"
    ]

packages = [
    'nirtools',
]

setup(
    name="nirtools",
    version=f"{__version__}",
    install_requires=install_requires,
    packages=packages,
)
