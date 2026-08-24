# causal-inference-pd

## requirements

- python 3.12
- packages: numpy, pandas, scipy, scikit-learn, openpyxl, tqdm

```
pip install numpy pandas scipy scikit-learn openpyxl tqdm
```

the required data files are in `data/` and are included in the repo.

## run

```
python main.py
```

optional arguments (used for testing before i accepted the final results... also for computation expense purposes):

```
--pc-reps INT      parkceleb repetitions (default 50)
--nv-reps INT      neurovoz repetitions (default 200)
--xfer-reps INT    transfer repetitions (default 200)
--cb-repeats INT   bootstrap repeats (default 10)
--reps INT         sets all of the above to one value
--seed INT         random seed (default set in old/causal.py)
```

results are written as csvs to `results/`.
