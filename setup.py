from setuptools import setup

setup(
    name="physlite_experiments",
    version="0.0",
    packages=["physlite_experiments"],
    install_requires=[
        "numpy",
        "h5py",
        "numba",
        "uproot>=4.0.1",
        "awkward>=1.1.0rc1",
        "pyarrow>=3",
        "coffea>=0.7",
    ],
    python_requires=">=3.5",
    author="Nikolai Hartmann",
    author_email="nihartma@cern.ch",
    description="Experiments with columnar data analysis with DAOD_PHYSLITE",
)
