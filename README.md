# TSTMotion: Training-free Scene-aware Text-to-motion Generation

<p align="left">
<!--     <a href='https://arxiv.org/abs/2210.09729'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://silvester.wang/HUMANISE/paper.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a> -->
    <a href='https://TSTMotion.github.io/TSTMotion.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

This repository is an official implementation of [TSTMotion](https://TSTMotion.github.io/TSTMotion.github.io/). The code is being cleaned and will be released soon.


## Folder Structure
```
├── datasets
│   ├── demo_scene
│   ├── HumanML3D
│   │   ├── new_joint_vecs
│   │   ├── new_joints
│   ├── prompt
│   ├── smplx
├── OmniControl
│   ├── glove
│   ├── t2m
│   ├── save
│   │   ├── omnicontrol_ckpt
│   │   │   ├── model_humanml3d.pt
├── scripts
├── utils
```

## Environment Setup
Our code is based on the [Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model), you may use the following command directly or refer to the [Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model) repository.
```
conda env create -f environment.yml
conda activate tstmotion
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```


## Citation
If you find TSTMotion useful for your work please cite:
```
@article{       ,
  author    = {},
  title     = {TSTMotion: Training-free Scene-aware Text-to-motion Generation},
  journal   = {},
  year      = {2024},
}
```
