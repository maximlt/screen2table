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

## V2

The command above doesn't work anymore because now the config file has to be included (it's not picked up automatically because it's not a Python file anymore). 

Using UPX doesn't reduce the size so much (from 35Mo to 25Mo) and leads to new errors.
```
pyinstaller --onefile --windowed --add-data "screen2table/configs.cfg;screen2table" --upx-dir=upx-3.95-win64 --upx-exclude "vcruntime140.dll" screen2table\screen2table.py
```

An attempt to debug the previous line. Run the EXE from the command line to get the DEBUG messages.
```
pyinstaller --onefile --add-data "screen2table/configs.cfg;screen2table" --upx-dir=upx-3.95-win64 --upx-exclude "vcruntime140.dll" --debug all screen2table\screen2table.py
```

The one that actually worked! The code had to be adapted to that config (PyInstaller and --onefile).
```
pyinstaller --onefile --windowed --add-data "screen2table/configs.cfg;screen2table" screen2table\screen2table.py
```

So the final one should be:
```
pyinstaller --distpath .\dist_pyinstaller -y --clean --onefile --windowed --add-data "screen2table/configs.cfg;screen2table" --version-file pyinstaller/file_version_info.txt screen2table\screen2table.py
```