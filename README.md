Installation
-

1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

1. `conda create --name dnn --file requirements.txt; conda activate dnn`

Basic Usage
-

- `./dnn gen_images`
- `./dnn`
- `./dnn default --gui=1`

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

- use `conda list -e > requirements.txt` to store dependencies

Credits
-

Darius,Xavier,nn arch writers, Pawan