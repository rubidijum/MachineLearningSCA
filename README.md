# MachineLearningSCA
Machine learning faculty course project. Using machine learning techniques for Side Channel Analysis.

## Project description

Code is divided between a couple of jupyter notebooks:

* `0_0_SCA_datasets_exploration.ipynb`: Introduction to Side Channel Attacks, non-ML techniques, motivation for ML based approach and datasets exploration.

* `0_1_DPA.ipynb` : Differential Power Analysis technique for extracting secrets from hardware.

* `0_2_CPA.ipynb` : Correlation Power Analysis tecnhique for extracting secrets form hardware.

* `1_0_SCA_MLP.ipynb` : Multilayer perceptrons for side channel attacks.

* `1_1_SCA_CNN.ipynb` : Convolutional neural networks for side channel attacks.

* `1_2_SCA_RNN.ipynb` : Recurrent neural networks for side channel attacks.

* `2_0_SCA_Model_Comparations.ipynb` : Conclusions


And utility python scripts used in notebooks:

* `AES.py` : Implement basic AES operations.

* `data_preparation.py` : Create and retrieve datasets.

* `training.py` : Training and evaluation helper functions.

## How to run:

### Setup the environment
---
In order to run code in this repository, python <version> and tensorflow <version> are needed.
You can use anaconda virtual environment manager to create virtual environment with all of the required packages and execute the code as normal jupyter notebook:

* Download Anaconda from [here](https://docs.conda.io/en/latest/miniconda.html) and install it.

* Start Anaconda prompt on Windows or just open a terminal on Linux.

* Create conda environment:

    `conda create -n env_name --file ml-sca.yaml`

* Activate newly created environment

    `conda activate env_name`
    
* Start jupyter notebook from repository root directory

    `cd <repository-path>`
    
    `jupyter notebook`

### Download data
---
Datasets and pretrained models can be downloaded from:

1) [SCAAML datasets](https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip) - Publicly available dataset of power traces.
2) [SCAAML models](https://storage.googleapis.com/scaaml-public/scaaml_intro/models.zip) - 
3) [ASCAD database - fixed key](https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip) - Data used in ASCAD paper.
4) [Trained CNN models](https://drive.google.com/file/d/1F1Ju0JOqwYjhIOejbE9-5SuhUwVdLkXV/view?usp=sharing)
5) [Trained MLP models](https://drive.google.com/file/d/11nqK43Gc1Rj7C3eHS_hiTtDB4xndE8BY/view?usp=sharing)
6) Training logs: [MLP](https://drive.google.com/file/d/1MBjH5ChTuFMZqtUB0Ep8hzKkc8AfMKMG/view?usp=sharing) [CNN](https://drive.google.com/file/d/1yjopZDvr9RLPdVC11ZBnRNumEBshNUGl/view?usp=sharing)
7) Hyperparameter tuning logs are too large to upload (~18GB)

NOTE: make sure to extract datasets to `data/` folder, models to `models/` folder and logs to `logs/` folder.

## References:

* [Study of Deep Learning Techniques for
Side-Channel Analysis and Introduction to
ASCAD Database](https://eprint.iacr.org/2018/053.pdf 'ASCAD')
* [Keras Tuner](https://keras.io/guides/keras_tuner/getting_started/ 'Keras Tuner')
* [Power Analysis Intro](https://www.youtube.com/watch?v=OlX-p4AGhWs&t=7s 'SCA intro')
* [Tuning neural networks for SCA](https://www.youtube.com/watch?v=uSpFfacjU4g&t=2146s 'Riscure NN tuning')
* [Deep learning SCA part 1](https://elie.net/blog/security/hacker-guide-to-deep-learning-side-channel-attacks-the-theory/#toc-7 'SCA DL theory')
* [Deep learning SCA part 2](https://elie.net/blog/security/hacker-guide-to-deep-learning-side-channel-attacks-code-walkthrough/ 'SCA DL code')
* [SCAAML github](https://github.com/google/scaaml/tree/1de561a95416f54d44b6fd18c79799064ea83163 'SCAAML github')
* [ASCAD github](https://github.com/ANSSI-FR/ASCAD 'ASCAD github')

## Citations:
    
 Project was inspired and greatly influenced by great work of the SCAAML team and amazing blog posts by Ellie Bursztein.
    
 ```bibtex
@online{bursztein2019scaaml,
  title={SCAAML:  Side Channel Attacks Assisted with Machine Learning},
  author={Bursztein, Elie and others},
  year={2019},
  publisher={GitHub},
  url={https://github.com/google/scaaml},
}
```
