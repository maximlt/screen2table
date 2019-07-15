# Pyinstaller mini guide

## Installing it

```
pip install pyinstaller
```

## Using it
Run this command from the root directory:
```
pyinstaller --onefile --windowed screen2table/screen2table.py --version-file pyinstaller/file_version_info.txt
```
This will create a `build` and a `dist` directory, as long with a `.spec` file.

A single executable can be found in the `dist` directory, that's the screen2table app and all its dependencies bundled into one binary file.

With `--windowed` running the executable would launch a console.

`--version-file pyinstaller/file_version_info.txt` grabs data from the version file to add them to the executable (version, name, copyright, etc.).