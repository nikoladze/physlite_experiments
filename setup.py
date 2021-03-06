from setuptools import setup

setup(
    name="physlite_experiments",
    version="0.0",
    packages=["physlite_experiments", "physlite_experiments.scripts"],
    package_data={"physlite_experiments" : ["data/*"]},
    install_requires=[
        "numpy",
        "h5py",
        "numba",
        "uproot>=4.0.1",
        "awkward>=1.1.0rc1",
        "pyarrow>=3",
        "coffea>=0.7",
        "aiohttp",
    ],
    python_requires=">=3.5",
    author="Nikolai Hartmann",
    author_email="nihartma@cern.ch",
    description="Experiments with columnar data analysis with DAOD_PHYSLITE",
)
