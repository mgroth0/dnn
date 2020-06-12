`conda create --name dnn --file requirements.txt; conda activate dnn`

Edit `selected_config()` at the top of `exp_config.py`

`python dnn.py --mode=FULL --muscle=LOCAL --gui=0`

DEV

`conda list -e > requirements.txt`