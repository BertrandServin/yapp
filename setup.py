from setuptools import setup
import yapp


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="yappgen",
    version=yapp.__version__,
    description="Yet Another Phasing Program",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://forgemia.inra.fr/bertrand.servin/yapp",
    author="Bertrand Servin",
    author_email="bertrand.servin@inrae.fr",
    license="LGPLv2.1",
    packages=["yapp"],
    python_requires="<3.10",
    install_requires=[
        "numpy >= 1.20.2",
        "scipy >= 1.6.2",
        "cyvcf2 >= 0.30.4",
        "zarr >= 2.7.0",
        "h5py >= 3.2.1",
        "fastphase >= 1.2",
        "numba >= 0.53.1",
        "pytoulbar2",
    ],
    scripts=["bin/fphtrain", "bin/yapp"],
    zip_safe=False,
)
