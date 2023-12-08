import setuptools


with open("readme.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='spideymaps',
    version='0.0.1',
    author='Daniel Foust',
    author_email='djfoust@umich.edu',
    description='For generating heat maps of single-molecule localization data in rod-shaped bacteria.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'shapely',
        'scikit-image',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    )
)