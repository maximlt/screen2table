import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="screen2table",
    version="0.3.0",
    description=(
        "A Windows GUI app to generate geometric data table"
        " by tracing shapes displayed on the screen."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/maximlt/screen2table",
    author="Maxime Liquet",
    author_email="maximeliquet@free.fr",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        'Operating System :: Microsoft :: Windows',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    packages=['screen2table'],
    package_data={'screen2table': ['configs.cfg']},
    python_requires='~=3.7',
    install_requires=[
        "matplotlib==3.1.1",
        "numpy==1.15.4",
        "pywin32==224",
        "pynput==1.4.4"
    ],
    entry_points={
        "console_scripts": [
            "screen2table=screen2table.screen2table:main",
        ]
    },
)
