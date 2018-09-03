# Bussell *et al.* 2018 Illustrative Epidemic Model

This project is provided as supplementary material to Bussell *et al.* 2018 (https://doi.org/10.1101/405746). We provide an implementation of the illustrative model described therein, as well as the simulation run data used to generate the figures in the paper. The model is a simple epidemic model incorporating risk and spatial structure. We also provide code to fit and optimise two approximate models, as described in the paper.

## Prerequisites

### BOCOP
We use the software package BOCOP v2.0.5 for optimisation of control on the approximate models. This must be installed on your machine to use this functionality. Installation instructions can be found on the BOCOP website (http://www.bocop.org/). We also provide a dockerfile that installs BOCOP for ease and/or reference.

The optimisation code for the two approximate models must be built first to provide a BOCOP executable.
```
cd path/to/RiskModelBOCOP
mkdir build
cmake -G "MSYS Makefiles" -DPROBLEM_DIR:PATH=path/to/RiskModelBOCOP path/to/BOCOP_Installation -DCMAKE_BUILD_TYPE=Release
```
And similarly for the space based optimisation folder

### Python
We use python v3.6.3 available from https://www.python.org/.

## File Descriptions

The following scripts are used to generate data and figures in the paper and can be used to reproduce the paper results. Note that many of these scripts take significant time to run due to the number of simulation realisations and the optimisations carried out. All other files provide additional code that is described within the file.

- `lag_time_analysis.py`  Used to generate data and figures for supplementary figure S7, comparing timing of deterministic and stochastic analogues
- `make_network.py`       Used to randomise the network setup and host risk groups
- `make_paper_figs.py`    This is used to generate most of the data and figures from the paper. Run with `-h` option for more info
- `profile_likelihoods.py`    Used to generate profile likelihood plots in supplementary material, verifying accuracy of the MLE fits. Supplementary figures S5 and S6
- `risk_split_scan.py`    Used to scan over control allocation to each risk group to find optimal constant split. Supplementary material figure S8
- `risk_switch_scan.py`   Used to scan over risk group switch time. Supplementary figure S11
