# SACuP: Sonar Image Augmentation with Cut and Paste Based DataBank for Semantic Segmentation
Code for the paper "SACuP: Sonar Image Augmentation with Cut and Paste Based DataBank for Semantic Segmentation"
### Abstract
> In this paper, we introduce Sonar image Augmentation with Cut and Paste based DataBank for semantic segmentation (SACuP), a novel data augmentation framework specifically designed for sonar imagery. Unlike traditional methods that often overlook the distinctive traits of sonar images, SACuP effectively harnesses these unique characteristics, including shadows and noise. SACuP operates on an object-unit level, differentiating it from conventional augmentation methods applied to entire images or object groups. Improving semantic segmentation performance while carefully preserving the unique properties of acoustic images is differentiated from others. Importantly, this augmentation process requires no additional manual work, as it leverages existing images and masks seamlessly. Our extensive evaluations, contrasting SACuP against established augmentation methods, unveil its superior performance, registering an impressive 1.10% gain in mean Intersection over Union (mIoU) over the baseline. Furthermore, our ablation study elucidates the nuanced contributions of individual and combined augmentation methods, such as cut and paste, brightness adjustment, and shadow generation, to model enhancement. We anticipate SACuP’s versatility in augmenting scarce sonar data across a spectrum of tasks, particularly within the domain of semantic segmentation. Its potential extends to bolstering the effectiveness of underwater exploration by providing high-quality sonar data for training machine learning models.
### Pipeline
![Pipeline](./figures/pipeline.png)
### Experiments
| Object | Baseline | TA | CutOut | CutMix | ObjectAug | Sim2Real | Ours |
|:------:|:--------:|:--:|:------:|:------:|:---------:|:--------:|:----:|
| background | 99.28 | 99.29 | 99.26 | 99.26 | 99.28 | 99.28 | 99.29 |
| bottle | 76.03 | 76.01 | 75.15 | 75.72 | 75.52 | 79.30 | 76.64 |
| can | 56.44 | 58.12 | 55.34 | 56.43 | 55.21 | 57.02 | 58.99 |
| chain | 63.48 | 63.44 | 62.00 | 61.83 | 62.35 | 62.96 | 64.25 |
| drink-carton | 73.75 | 74.65 | 72.44 | 74.31 | 73.31 | 74.30 | 75.95 |
| hook | 67.73 | 68.87 | 68.41 | 67.62 | 68.18 | 68.47 | 69.41 |
| propeller | 73.19 | 74.37 | 72.88 | 73.67 | 74.85 | 73.03 | 74.89 |
| shampoo-bottle | 78.07 | 79.91 | 78.18 | 78.51 | 79.47 | 78.88 | 78.61 |
| standing-bottle | 79.83 | 80.00 | 79.66 | 78.90 | 82.67 | 79.54 | 81.23 |
| tire | 88.00 | 87.64 | 87.63 | 87.49 | 87.65 | 87.61 | 87.92 |
| valve | 58.11 | 58.33 | 58.27 | 58.95 | 58.36 | 58.47 | 59.56 |
| wall | 87.74 | 88.75 | 88.24 | 88.07 | 88.38 | 88.31 | 88.17 |
| mIoU | 75.14 | 75.78 | 74.79 | 75.06 | 75.44 | 75.35 | **76.24** |
## Usage
### Setup Workspace
Clone this Git repository.
```bash
git clone https://github.com/AIRLABkhu/SACuP.git
cd SACuP
```
### Train & Test
```bash
python main.py
```
<!--
### Augmentation
```bash

```
### Training
```bash

```
### Testing
```bash

```
-->
