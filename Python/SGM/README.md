# SGM

This project has an implementation of [Semi-Global Matching](https://core.ac.uk/download/pdf/11134866.pdf) that utilizes the [Census cost metric](https://www.cs.middlebury.edu/~schar/papers/evalcosts-pami08.pdf) on scenes from the Middlebury dataset.

## Instructions
Tested on Ubuntu 20.04.6  and Ubuntu 22.04.4

1. Open up terminal in the main directory where the ``environment.yaml`` file is located.
2. Run ``conda env create -f environment.yaml`` and follow the conda prompts for environment setup.
3. When the environment is done, activate it with ``conda activate sgbm_compvis``.
4. Run the project with the command ``python main.py``. It might take several minutes.
5. Output disparity images and relevant ground truth images should be saved in `./output/`.
6. Statistics and metrics for each part (baseline and SGM) are saved in `arbitrary.csv`.
7. To run on an arbitrary image, go to "main.py" and fiddle with ``run_arbitrary()`` to edit the paths ``"imL.png``,``imR.png``,``ground.png`` appropriately according to where that image is relative to the directory ( I would just replace the example images I have). NB: I have attempted to support ``.pfm`` extensions here, but loading everything as a ``.png`` might be best.



