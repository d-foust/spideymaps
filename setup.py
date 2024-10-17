import setuptools


with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='spideymaps',
    version='0.1.2',
    author='Daniel Foust',
    author_email='djfoust@umich.edu',
    description='For generating heat maps of single-molecule localization data in rod-shaped bacteria.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        'h5py>=3.0.0',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'shapely>=2.0.0',
        'scikit-image',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)