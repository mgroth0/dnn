Installation
-
1. git clone --recurse-submodules https://github.com/mgroth0/dnn
1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
1. `conda update conda`
1. `conda create --name dnn --file requirements.txt` (requirements.txt is currently not working, TODO)
1. might need to separately `conda install -c mgroth0 mlib-mgroth0`-- When updating, use `conda install --file requirements.txt;`
1. `conda activate dnn`


Usage
-
- ./dnn
- Generate some images, train/test a model, run analyses, and generate plots. Tested on Mac, but not yet on linux/Windows.

- `./dnn -cfg=gen_images --INTERACT=0`
- `./dnn -cfg=test_one --INTERACT=0`

The second command will fail with a Mathematica-related error, but your results will be saved in `_figs`.

TODO: have to also consider running and developing other executables here: human_exp_1 and human_analyze




Configuration
-
-MODE: (default = FULL) is a string that can contain any combination of the following (example: "CLEAN JUSTRUN")
- CLEAN
- JUSTRUN
- GETANDMAKE
- MAKEREPORT

Edit [cfg.yml]() to save configuration options. Feel free to push these.

If there is anything hardcoded that you'd like to be configurable, please submit an issue.

Testing
-
todo

Development
-
- TODO: have separate development and user modes. Developer mode has PYTHONPATH link to mlib and instructions for resolving and developing in ide in parallel. User mode has mlib as normal dependency. might need to use `conda uninstall mlib-mgroth0 --force`. Also in these public readmes or reqs.txt I have to require a specific mlib version
- ./dnn build


Credits
-
Darius, Xavier, Pawan

heuritech, raghakot, joel