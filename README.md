Installation
-

1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. `conda update conda`

3. `conda config --add channels conda-forge`

4. `conda create --name dnn --file requirements.txt; pip install -r reqs_pip.txt`  
-- When updating, use `conda install --file requirements.txt; pip install -r reqs_pip.txt`

5. `conda activate dnn`

Basic Usage
-

- `./dnn -cfg=gen_images`
- `./dnn -cfg=test_one`

Configuration
-

Mode: (default = FULL) is a string that can contain any combination of the following (example: "CLEAN JUSTRUN")
- CLEAN
- JUSTRUN
- GETANDMAKE
- MAKEREPORT

Edit [cfg.yml]() to save configuration options. Feel free to push these.

If there is anything hardcoded that you'd like to be configurable, please submit an issue.'

Development
- 

- use `conda list -e > requirements.txt; sed -i '' '/pypi/d' requirements.txt` to store dependencies

Credits
-

Darius,Xavier,nn arch writers, Pawan