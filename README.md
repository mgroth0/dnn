Installation
-

1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. `conda update conda`


3. `conda config --add channels conda-forge` (might not be required)

4. `conda create --name dnn --file requirements.txt; pip install -r reqs_pip.txt`  
-- When updating, use `conda install --file requirements.txt; pip install -r reqs_pip.txt`

5. `conda activate dnn`

Basic Usage
-

Generate some images, train/test a model, run analyses, and generate plots. Tested on Mac, but not yet on linux/Windows.

- `./dnn -cfg=gen_images --INTERACT=0`
- `./dnn -cfg=test_one --INTERACT=0`

The second command will fail with a Mathematica-related error, but your results will be saved in `_figs`.

Configuration
-

-MODE: (default = FULL) is a string that can contain any combination of the following (example: "CLEAN JUSTRUN")
- CLEAN
- JUSTRUN
- GETANDMAKE
- MAKEREPORT

Edit [cfg.yml]() to save configuration options. Feel free to push these.

If there is anything hardcoded that you'd like to be configurable, please submit an issue.

Development
- 

- use `conda list -e > requirements.txt; sed -i '' '/pypi/d' requirements.txt` to store dependencies.
- There are also a couple of pip dependencies manually written in reqs_pip.txt, since these cannot be found through conda

Credits
-

- Darius, Xavier, Pawan
- heuritech, raghakot, joel
