# Sparse Coding PyTorch

This code implements sparse coding in PyTorch with the positive-only option. For the positive-only option, I only constraint the sparse coefficients to be non-negative. This choice is related but different from non-negative sparse coding or non-negative matrix factorization. The optimization solver used in this code is ISTA and FISTA. To demo the code, whitened natural images are adapted from: http://www.rctn.org/bruno/sparsenet/

### A Sample of the Learned Dictionary
<p align="center">
  <img src="./figures/sparse_coding.png" width=600>
</p>

In order to use this code, python 3 and PyTorch 0.4 or above is required. Please follow the steps in the following Jupyter notebook:

```bash
sparse_coding_torch_Demo.ipynb
```

The following are some useful references:

### Sparse Coding
```bash
@article{olshausen1996emergence,
  title={Emergence of simple-cell receptive field properties by learning a sparse code for natural images},
  author={Olshausen, Bruno A and Field, David J},
  journal={Nature},
  volume={381},
  number={6583},
  pages={607},
  year={1996},
  publisher={Nature Publishing Group}
}
```
```bash
@inproceedings{olshausen2013highly,
  title={Highly overcomplete sparse coding},
  author={Olshausen, Bruno A},
  booktitle={Human Vision and Electronic Imaging XVIII},
  volume={8651},
  pages={86510S},
  year={2013},
  organization={International Society for Optics and Photonics}
}
```

### FISTA Algorithm
```bash
@article{beck2009fast,
  title={A fast iterative shrinkage-thresholding algorithm for linear inverse problems},
  author={Beck, Amir and Teboulle, Marc},
  journal={SIAM journal on imaging sciences},
  volume={2},
  number={1},
  pages={183--202},
  year={2009},
  publisher={SIAM}
}
```

### Postive-Only Sparse Coding
```
@inproceedings{hoyer2002non,
  title={Non-negative sparse coding},
  author={Hoyer, Patrik O},
  booktitle={Proceedings of the 12th IEEE Workshop on Neural Networks for Signal Processing},
  pages={557--565},
  year={2002},
  organization={IEEE}
}
```

### The Sparse Manifold Transform
```
@inproceedings{DBLP:conf/nips/ChenPO18,
  author    = {Yubei Chen and
               Dylan M. Paiton and
               Bruno A. Olshausen},
  title     = {The Sparse Manifold Transform},
  booktitle = {Advances in Neural Information Processing Systems 31: Annual Conference
               on Neural Information Processing Systems 2018, NeurIPS 2018, December
               3-8, 2018, Montr{\'{e}}al, Canada},
  pages     = {10534--10545},
  year      = {2018}
}
```

