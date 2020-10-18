# msMINRES-CIQ Experiments

Code to reproduce the experiments in
"Fast Matrix Square Roots with Applications to Gaussian Processes and Bayesian Optimization"
by Geoff Pleiss, Martin Jankowiak, David Eriksson, Anil Damle, and Jacob R. Gardner (NeurIPS 2020).

** N.B. ** This code requires a currently-unreleased feature in GPyTorch. The feature will be added shortly.

## System Requirements:
 - Python 3.7
 - PyTorch 1.6
 - GPyTorch 1.3
 - NumPy 
 - scipy.cluster
 - scikit-learn
 - tqdm
 - pandas
 - a GPU


## SVGP experiments
These are located in the `svgp/` folder.
Explanations for the command line args can be found in `uci_regression.py`.

```sh
# for msMINRES-CIQ SVGP
python uci_regression.py -d 3droad  -vs ciq --likelihood gaussian  --num-ind 2000 --batch-size 256 
python uci_regression.py -d precip  -vs ciq --likelihood studentt  --num-ind 2000 --batch-size 256 -lr 0.005 -vlr 0.005
python uci_regression.py -d covtype -vs ciq --likelihood bernoulli --num-ind 2000 --batch-size 512 

# for Cholesky SVGP
python uci_regression.py -d 3droad  -vs standard --likelihood gaussian  --num-ind 2000 --batch-size 256 
python uci_regression.py -d precip  -vs standard --likelihood studentt  --num-ind 2000 --batch-size 256 -lr 0.005 -vlr 0.005
python uci_regression.py -d covtype -vs standard --likelihood bernoulli --num-ind 2000 --batch-size 512 
```


## Super-resolution experiments
These experiments rely on code in the `super_resolution` folder.
They require the additional packages
- Open CV2
- Kornia
- TorchVision

```sh
python sr.py lion160.png 5 2.5  # Replace with lion96.png if this doesn't fit on your GPU
```
