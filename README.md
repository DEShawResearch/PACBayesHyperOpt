Efficient Hyperparameter Optimization By Way of PAC-Bayes Bound Minimization
====================================================================

This repository contains scripts accompanying our manuscript 'Efficient Hyperparameter
Optimization By Way of PAC-Bayes Bound Minimization.' The
scripts demonstrate an implementation of the proposed method and can be used
to reproduce variants of the figures which appear in the manuscript (the default
values of some settings in these scripts, such as the number of optimization steps and
number of replicates, are lower than those used to produce the figures in the manuscript).

Prerequisites
-------------
These scripts are implemented in Python with PyTorch, and were developed on Linux. In
order to run them you'll need at least the following software

1. Python 3.7 or greater (https://www.python.org/downloads/release/python-376/)
2. NumPy 1.15 or greater (https://pypi.org/project/numpy/)
3. PyTorch 1.2 or greater (https://pytorch.org/get-started/locally/)
4. CUDA 10 and a sufficiently recent NVIDIA GPU. Our experiments were performed on NVIDIA
   GTX 1080 and 980 Ti GPUs.
5. Higher 0.5.1 (https://github.com/facebookresearch/higher)
6. For plotting, a relatively recent version of matplotlib, seaborn and pandas.

Contents
--------
* The `freedman` directory contains experiments related to Freedman's paradox, a feature
  selection problem. The script `freedman.py` creates Freedman's paradox training,
  validation and test data sets and runs forward selection with our proposed regularizer.
  The script `freedman_naive.py` consumes the same data set and runs naive forward
  selection using only the validation set error. A visual comparison of the results of
  the two methods can be produced with the script `plot-comparison.py`, which creates
  the plots `regularized_objective.pdf` and `unregularized_objective.pdf` (copies of
  which are included for reference).
* The `weight_decay` directory contains experiments related to hyperparameter
  optimization for neural network image classifiers with a separate per-parameter weight
  decay hyperparameter. The models are fit to small subsets of of MNIST or CIFAR10 using
  a linear classifier, ResNet-18, or ResNet-34. Because of the small number of
  training and validation data points and the large number of hyperparameters, it is 
  very easy to overfit to the validation set. The script `weight_decay.py` demonstrates 
  Algorithm C.1/C.2 from the paper. The script `weight_decay_higher.py` uses a more 
  computationally demanding approximation to the hyperparameter gradient with
  extended truncated backpropagation through multiple inner steps (see Appendix C).

References
----------
Krizhevsky, A. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 04 2009. URL http://www.cs.toronto.edu/-kriz/learning-features-2009-TR.pdf.  
LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998. URL http://yann.lecun.com/exdb/mnist/.
