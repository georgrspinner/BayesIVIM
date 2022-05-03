# BayesIVIM
Code for paper:

Georg Ralph Spinner, Christian Federau and Sebastian Kozerke.
Bayesian inference using hierarchical and spatial priors for intravoxel incoherent motion MR imaging in the brain: Analysis of cancer and acute stroke.
Medical Image Analysis (73), October 2021

https://doi.org/10.1016/j.media.2021.102144


This repository contains realistic numerical brain MRI phantoms with pathological lesions (cancer, acute stroke), least-squares based IVIM fits and routines for Bayesian inference - including the proposed method.

Try demo.m to test it.

![grafik](https://user-images.githubusercontent.com/72972409/162079761-82bfce23-e9a3-4318-b07d-6baff36c6565.png)


Non-essential dependencies:
- Numerical brain MRI phantom: http://bigwww.epfl.ch/algorithms/mriphantom/ (requires compilation), corresponding paper: https://doi.org/10.1109/TMI.2011.2174158; only necessary if numerical phantoms need to be re-created or modified
- Python and packages: only necessary for deep-learning-based IVIMNET fits, which will be skipped if Matlab can not execute the Python calls correctly. Required packages are installed automatically. A Python script to perform IVIMNET (https://github.com/oliverchampion/IVIMNET, corresponding paper: https://doi.org/10.1002/mrm.28852) fits is included.
- Image registration: pTVreg (https://github.com/visva89/pTVreg, corresponding paper: https://doi.org/10.1109/TMI.2016.2610583) is used for potential in-vivo data (not included), but other registration tools might work as well.

The code is developed by Georg Ralph Spinner (georg.spinner@zhaw.ch) at [Cardiac Magnetic Resonance group](http://www.cmr.ethz.ch/), Institute for Biomedical Engineering, ETH Zurich and University of Zurich

![grafik](https://user-images.githubusercontent.com/72972409/162081945-817ccb84-e1d5-4e06-8449-9e14085823d9.png)
![grafik](https://user-images.githubusercontent.com/72972409/162081219-26281bc8-53a3-4573-bad1-675fb4f9c41c.png)
