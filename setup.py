from setuptools import setup, find_packages

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
    version="0.0.1",
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