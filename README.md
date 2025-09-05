## rm_classifier
This package provides tools for cluster identification and classification tasks, related to the monitoring of radio antennas.
It includes functionalities for data preprocessing, model training, evaluation, and visualization.

## Installation

### Method 1: Conda Environment + Pip (Recommended)

<pre>
```bash
git clone https://github.com/uhgtb/rm_classifier.git
cd rm_classifier
conda env create -f environment.yml
conda activate rm_classifier
pip install -e .
``` </pre>

### Method 2: Pip (might not run in some python versions)
<pre>
```bash
pip install git+https://github.com/uhgtb/rm_classifier.git
``` </pre>

### Stable version
If any package compatibility isuues with newer versions occur, create an environment with stable package version by running

<pre>
```bash
git clone https://github.com/uhgtb/rm_classifier.git
cd rm_classifier
conda env create -f stable_environment.yml
conda activate stable_rm_classifier
pip install -e .
``` </pre>

## Documentation
Sphinx documentation in the /docs/_build/html/index.html folder or can be found at [readthedocs](https://rm-classifier.readthedocs.io/).
