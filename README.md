<h1 align="center">White blood cell detection</h1>

### Detect white blood cells in blood smear images

Put the data in a folder called `data` in the root directory of this repo.
The script will generate output in the `output` folder.

main.py - extract wbc from ndpi file
put the model for classified ROI and also model for wbc detection in the same folder, and rename the model name in main.py to extract wbc from ndpi file.

extract.py - extract wbc from jpg file
put the model for wbc detection in the same folder, and rename the model name in extract.py.

The ultralytics package is ver.8.0.20, the newer version might not work with some functions.
