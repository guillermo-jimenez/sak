import codecs
import os.path

from setuptools import setup, find_packages

"""Reading version as proposed in https://packaging.python.org/guides/single-sourcing-package-version/"""
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()
def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="sak",
    description="Swiss Army Knife - General utilitiles for signal and image processing",
    url="https://github.com/gjimenez/sak",
    author="Guillermo Jimenez-Perez",
    author_email="<guillermo@jimenezperez.com>",
    # Needed to actually package something
    packages=find_packages(),
    # Needed for dependencies
    install_requires=["numpy", "scipy", "pandas", "wfdb", "dill", 
                      "torch", "torchvision", "tqdm", "scikit-image", 
                      "networkx", "defusedxml"],
    # *strongly* suggested for sharing
    version=get_version("sak/__init__.py"),
    # The license can be anything you like
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    # We will also need a readme eventually (there will be a warning)
    long_description=open("README.txt").read(),
    long_description_content_type="text/markdown",
)