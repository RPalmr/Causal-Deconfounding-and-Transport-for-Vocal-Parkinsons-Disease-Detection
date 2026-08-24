# causal-sampler
``causal-sampler`` is a Python package that integrates multiple causal (re-)sampling techniques, e.g., **causal bootstrapping (CB)** and **causally weighted Gaussian Mixture Models (CW-GMMs)**, offering a standardized and user-friendly high-level pipeline and interfaces. 

By performing causal resampling, the causally biased observational data can be **deconfounded** in a way that approximates the experimental data that was collected from a well-controlled environment, such as Randomized Controlled trials. Given this, the resampled deconfounded data should benefit many downstream applications, e.g., enable the "causal-blind" machine learning to learn the intended causal relationship between the high-dimensional features and the prediction target variable rather than potentially biased and spurious correlations.

<p align="center">
  <img src="https://github.com/JianqiaoMao/causal-sampler/blob/main/figures/causal_sampler.png" width="1000">
</p>

## Example of "background brightness" confounding in **MNIST** dataset

<p align="center">
  <img src="https://github.com/JianqiaoMao/causal-sampler/blob/main/figures/backdoor_graph.png" width="300">
  <br><sub><em>Figure 1. Backdoor confounding. </em></sub>
</p>

In a backdoor confounding setting, an existing confounder acting as a common cause of the cause ($Y$) and effect ($X$) variables of interest may lead to so-called "selection bias". A dataset collected in such an environment can be severly causally biased due to the confounder $U$. When a machine leanring model which is blind to the backend causal relationships between variables is trained with the confounded observational dataset, it is exposed to risks of learning unreliable or even spurious associations between the prediction target and the features. A simple and intuitive example is as below:

<p align="center">
  <img src="https://raw.githubusercontent.com/JianqiaoMao/causal-sampler/main/figures/background_MNIST.png" width="800" alt="MNIST background">
  <br><sub><em>Figure 2. Example digits from the confounded (a) and non-confounded (b) background-MNIST datasets. In (a), background brightness is manipulated so that it is a confounding factor with digit class (e.g., "6" is brighter than "2"); in (b), the brightness-digit association is randomized. </em></sub>
</p>

In the confounded dataset (Fig. 2 (a)), images of digit "6" tend to have brighter backgrounds than images of digit "2", but this confounding effect is not present for the non-confounded dataset (Fig. 2 (b)). Here, we consider the digit categories ("2" or "6" as the cause variable $Y$, the handwriting images as the effeect variable $X$ and the background brightness is the confounder $U$ that acts as common cause that connects the backdoor path between the images and categories.

A standard supervised classifier trained on the confounded MNIST dataset is likely to make predictions based on the input image's average brightness rather than the handwriting digit's actual shape because the brightness feature is strongly associated with the label. Thus, any supervised learning algorithm will use this spurious brightness information to maximize prediction accuracy. However, this brightness information is not what we expect the classifier to learn; should the predictor be applied to data without this brightness confounder, it would be of no use, despite the apparently high out-of-sample accuracy of the predictor.

The ``causal-sampler`` package provides multiple causal resampling techniques that can utilize the confounded dataset to generate deconfounded dataset like how Fig.2 (b) shows.

## Citing

Please use one of the following to cite the code of this repository.

```
@article{mao2024mechanism,
  title={Mechanism learning: Reverse causal inference in the presence of multiple unknown confounding through front-door causal bootstrapping},
  author={Mao, Jianqiao and Little, Max A},
  journal={arXiv preprint arXiv:2410.20057},
  year={2024}
}

@article{little2019causal,
  title={Causal bootstrapping},
  author={Little, Max A and Badawy, Reham},
  journal={arXiv preprint arXiv:1910.09648},
  year={2019}
}
```

## Installation and getting started

We currently offer seamless installation with `pip`. 

Simply:
```
pip install causal-sampler
```

Alternatively, download the current distribution of the package, and run:
```
pip install .
```
in the root directory of the decompressed package.

To import the high-level interfaces:
```python
import causal_sampler.pipeline as cs_pipe
```

The current version provides two causal sampling techniques:

- Causal Bootstrapping [1]: ``CausalBootstrapSampler``
- Causally Weighted Gaussian Mixture Model [2]: ``CausalGMMSampler``

For more detailed demonstration, please see: [Demo](https://github.com/JianqiaoMao/causal-sampler/tree/main/Demo).

## Update plan

- Causal Dirichlet Process Mixture Models
    - [x] source code development
    - [x] Toy example experiments
    - [ ] Interface design and development
- Causal-aware Markov chain Monte Carlo-based Block Gibbs sampling
    - [x] source code development
    - [x] Toy example experiments
    - [ ] Interface design and development
- Denoising Diffusion Probabilistic Models (DDPM) based causal sample generation
    - [x] source code development
    - [ ] Toy example experiments
    - [ ] Interface design and development

## Reference

[1] Mao, Jianqiao, and Max A. Little. "Mechanism learning: Reverse causal inference in the presence of multiple unknown confounding through front-door causal bootstrapping." arXiv preprint arXiv:2410.20057 (2024).

[2] Little, Max A., and Reham Badawy. "Causal bootstrapping." arXiv preprint arXiv:1910.09648 (2019).

[3] Mao, Jianqiao. "mechanism-learn" Github repo. https://github.com/JianqiaoMao/mechanism-learn, https://zenodo.org/records/17306651.




