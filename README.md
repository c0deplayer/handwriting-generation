# Handwriting Generation
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

## Requirements
```
    Python>=3.8,
    gradio>=4.3.0,
    torch>=2.1.1,
    torchvision>=0.16.1,
    torchaudio>=2.1.1,
    lightning[extra]>=2.1.1,
    torchmetrics>=1.2.0,
    einops>=0.7.0,
    neptune>=1.8.3,
    dataclass-wizard>=0.22.2,
    setuptools>=68.2.2,
    h5py>=3.10.0,
    diffusers[torch]>=0.23.0,
    potracer>=0.0.4,
    clean-fid>=0.1.35
```

## Datasets & Pre-processing

Download the IAM Dataset and IAM Online Dataset from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
and https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database, respectively.
Place them in the `raw_data/IAMDB` and `raw_data/IAMonDB` folders, respectively.

Then, to preprocess the dataset and save it to an H5 file, simply run the following command:

```
python3 prepare_data.py -c {RNN,Diffusion,LatentDiffusion}
```

## Training from scratch

To train a diffusion model, run the following command:

```
python3 train.py -c Diffusion
```

## Generate handwriting

To generate handwriting run the following command:

```
python3 synthesize.py -c LatentDiffusion -t "the quick brown fox jumps" -w 64
```

## Full sampling

WIP

## Commands

### prepare_data.py

```
prepare_data.py [-h] -c {RNN,Diffusion,LatentDiffusion} [-cf CONFIG_FILE]

options:
  -h, --help            show this help message and exit
  -c {RNN,Diffusion,LatentDiffusion}, --config {RNN,Diffusion,LatentDiffusion}
                        Type of model
  -cf CONFIG_FILE, --config-file CONFIG_FILE
                        Filename for configs

```

### train.py

```
train.py [-h] -c {RNN,Diffusion,LatentDiffusion} [-cf CONFIG_FILE] [-r] [-n]

options:
  -h, --help            show this help message and exit
  -c {RNN,Diffusion,LatentDiffusion}, --config {RNN,Diffusion,LatentDiffusion}
                        Type of model
  -cf CONFIG_FILE, --config-file CONFIG_FILE
                        Filename for configs
  -r, --remote          Flag indicating whether the model will be trained on a server with dedicated
                        GPUs, such as the A100
  -n, --neptune         Flag for using NeptuneLogger
```

### synthesize.py

```
synthesize.py [-h] -c {RNN,Diffusion,LatentDiffusion} [-cf CONFIG_FILE] [-t TEXT] [-w WRITER]
                     [--color COLOR] [-s STYLE_PATH]

options:
  -h, --help            show this help message and exit
  -c {RNN,Diffusion,LatentDiffusion}, --config {RNN,Diffusion,LatentDiffusion}
                        Type of model
  -cf CONFIG_FILE, --config-file CONFIG_FILE
                        Filename for configs
  -t TEXT, --text TEXT  Text to generate
  -w WRITER, --writer WRITER
                        Writer style. If not provided, the default writer is selected randomly
  --color COLOR         Handwriting color. If not provided, the default color is black
  -s STYLE_PATH, --style_path STYLE_PATH
                        Filename for style. If not provided, the default style is selected randomly
```

### full_sample.py

```
full_sample.py [-h] -c {Diffusion,LatentDiffusion} [-cf CONFIG_FILE] [--strict]

options:
  -h, --help            show this help message and exit
  -c {RNN,Diffusion,LatentDiffusion}, --config {RNN,Diffusion,LatentDiffusion}
                        Type of model
  -cf CONFIG_FILE, --config-file CONFIG_FILE
                        Filename for configs
  --strict              Strict mode for a dataset that excludes OOV words
```

## References

1. [labml.ai Annotated PyTorch Paper Implementations](https://nn.labml.ai/)
2. [WordStylist: Styled Verbatim Handwritten Text
   Generation with Latent Diffusion Models](https://arxiv.org/pdf/2303.16576.pdf)
3. [Diffusion models for Handwriting Generation](https://arxiv.org/pdf/2011.06704v1.pdf)
4. [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf)
5. [High-Resolution Image Synthesis with Latent Diffusion Models
   (A.K.A. LDM & Stable Diffusion)](https://arxiv.org/pdf/2112.10752.pdf)
6. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
7. [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
8. [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202.pdf)
9. [IAM Handwriting Database & IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases)
10. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597v1.pdf)
11. [Compvis/stable-diffusion · hugging face](https://huggingface.co/CompVis/)
12. [Semi-Parametric Neural Image Synthesis](https://arxiv.org/pdf/2204.11824.pdf)
13. [Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network](https://arxiv.org/pdf/1808.03314.pdf)
14. [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://arxiv.org/pdf/2209.00796.pdf)
15. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
16. [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)
17. [Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101.pdf)
18. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
