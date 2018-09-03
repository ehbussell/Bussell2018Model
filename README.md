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
