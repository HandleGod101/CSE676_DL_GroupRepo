# Pytorch Implementation of CleanUNet

We use [CleanUNet](https://github.com/NVIDIA/CleanUNet) to implement this project. CleanUNet is a causal speech denoising model on the raw waveform. For more information please look at this paper:[Speech Denoising in the Waveform Domain with Self-Attention](https://arxiv.org/abs/2202.07790) For all the implementation details, please follow their instructions. We only include our innovation here.

## Datasets

- [Microsoft DNS 2020](https://arxiv.org/ftp/arxiv/papers/2005/2005.13981.pdf) dataset. The dataset, pre-processing codes, and instruction to generate training data can be found in [this link](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master). 

  [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html) dataset. This dataset contains 10 different classes of audio. You can directly download dataset using the link. Since this dataset only provide clean data, we manually add 10% gaussian noise to it.

## Training

Training step follows the [CleanUNet](https://github.com/NVIDIA/CleanUNet). 8GPUs are used for training. It took 12 hours in NVIDIA RTX A6000. We use linear warmup and cosine annealing for the learning rate.

## Evaluation

We use PESQ and STOI to evaluation the denoise result in both two datasets. Compare the result from network and traditional denoise filter (high pass, low pass etc.) in UrbanSound8k dataset.

1 GPU is used for evaluation.


## References

The code structure and distributed training are adapted from [WaveGlow (PyTorch)](https://github.com/NVIDIA/waveglow) (BSD-3-Clause license). The ```stft_loss.py``` is adapted from [ParallelWaveGAN (PyTorch)](https://github.com/kan-bayashi/ParallelWaveGAN) (MIT license). The self-attention blocks in ```network.py``` is adapted from [Attention is all you need (PyTorch)](https://github.com/jadore801120/attention-is-all-you-need-pytorch) (MIT license), which borrows from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) (MIT license). The learning rate scheduler in ```util.py``` is adapted from [VQVAE2 (PyTorch)](https://github.com/rosinality/vq-vae-2-pytorch) (MIT license). Some utility functions are borrowed from [DiffWave (PyTorch)](https://github.com/philsyn/DiffWave-Vocoder) (MIT license) and [WaveGlow (PyTorch)](https://github.com/NVIDIA/waveglow) (BSD-3-Clause license).
