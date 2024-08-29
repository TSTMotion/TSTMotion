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

This repository is an official implementation of [TSTMotion](https://TSTMotion.github.io/TSTMotion.github.io/) <b>BASELINE</b>. The <b>COMPLETE CODE</b> is being cleaned and will be released soon.


## Folder Structure
```
├── datasets
│   ├── demo_scene
│   ├── HumanML3D
│   │   ├── new_joint_vecs
│   │   ├── new_joints
│   ├── prompt
│   ├── smplx
│   │   ├── SMPLX_NEUTRAL.npz
│   │   ├── SMPLX_NEUTRAL.pkl
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

### 1. Setup Conda:
```
conda env create -f environment.yml
conda activate tstmotion
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

### 2. Download Dependencies:
The results should be placed as shown in Folder Structure, including glove,t2m and smplx.
```
cd OmniControl
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### 3. Download Dataset:
Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D), then copy the results as shown in Folder Structure.

### 4. Download Checkpoint:
The results should be placed as shown in Folder Structure, including [model_humanml3d.pt](https://drive.google.com/file/d/1oTkBtArc3xjqkYD6Id7LksrTOn3e1Zud/view).
```
cd /OmniControl/save
gdown --id 1oTkBtArc3xjqkYD6Id7LksrTOn3e1Zud
unzip omnicontrol_ckpt.zip -d .
```


## Acknowledgements

Some codes are borrowed from [MDM](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file), [HUMANISE](https://github.com/Silverster98/HUMANISE), [OmniControl](https://github.com/neu-vi/OmniControl).

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
