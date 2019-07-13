import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="screen2table",
    version="0.2.0",
    description=(
        "A Windows GUI app to generate geometric data table"
        " by tracing shapes displayed on the screen."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/maximlqt/screen2table",
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
    install_requires=["matplotlib", "pywin32", "pynput"],
    entry_points={
        "console_scripts": [
            "screen2table=screen2table.screen2table:main",
        ]
    },
)
