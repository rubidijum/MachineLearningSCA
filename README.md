# MachineLearningSCA
Machine learning faculty course project. Using machine learning techniques for Side Channel Analysis.

## How to run:

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

Datasets and pretrained models can be downloaded from:

1) [SCAAML datasets](https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip)
2) [SCAAML models](https://storage.googleapis.com/scaaml-public/scaaml_intro/models.zip)
3) [Trained models]()
4) [Logs]()

NOTE: make sure to extract datasets to `data/` folder.

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
