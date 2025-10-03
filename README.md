
# Explainable Deep Learning with Grad-CAM Variants
###### this README.md was created with the help of Chatgpt5 at 9:00pm on 10/2/25


## Overview
This project explores **explainable AI (XAI)** techniques applied to face images.  
We used convolutional neural network models with **GradCAM, GradCAM++**, and **LayerCAM** to visualize which regions of an image drive the model’s attention.  
The goal was to evaluate **whether the model focuses on appropriate features (eyes, nose, face structure)** or misleading cues (mouth, background, expressions), and to consider fairness implications.

---

## Dataset
We used images from the **[FairFace dataset](https://github.com/dchen236/FairFace/tree/master)**, a balanced face attribute dataset for race, gender, and age.  

**Citation:**  
Karkkainen, K., & Joo, J. (2021).  
*FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation.*  
In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1548-1558).  
[[Paper Link]](https://openaccess.thecvf.com/content/WACV2021/papers/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.pdf)

BibTeX:
```bibtex
@inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021},
  pages={1548--1558}
}
````


## Methods

1. **Face Detection & Preprocessing**

   * Used MTCNN to crop and align faces.
   * Optional step: applied Gaussian blur to the background while keeping the central face sharp.

2. **Grad-CAM Variants**

   * **GradCAM**: Standard gradient-based class activation mapping.
   * **GradCAM++**: Enhanced version providing sharper maps.
   * **LayerCAM**: Layer-wise refinement for improved localization.

3. **Focus Scoring**

   * Implemented a simple metric to quantify whether activations were centered on the face (higher = more focused).

---

## Results

### Cropped Faces

* The model frequently emphasized the **mouth and teeth**, especially when subjects were smiling.
* This suggests over-reliance on expressions rather than consistent identity features.
* Some groups (e.g., Black, Latino images) showed particularly strong mouth activations.

### Background-Blurred Faces

* Attention shifted more evenly across **eyes, nose, and forehead**, reducing the bias toward the mouth.
* This preprocessing step improved focus consistency across all groups.
* However, residual mouth emphasis remained in some cases.

---

## Interpretation

* **Appropriate Cues**: The blurred-background setup encouraged attention to stable face features.
* **Surprising/Misleading Cues**: Heavy reliance on smiles in cropped images was unexpected and potentially misleading.
* **Fairness Implications**: If deployed in sensitive domains (e.g., identity verification, hiring systems), such biases could unfairly impact groups whose images differ in expression.
* **Limitations**:

  * Small subset of FairFace used (not representative).
  * CAMs highlight correlations, not causation.
  * Only a single pretrained model was used, not fine-tuned for fairness.

---

## Significance

Explainability tools like GradCAM are essential for:

* **Auditing model fairness**
* **Identifying hidden biases**
* **Building trust in AI systems**

In domains like face recognition, healthcare, or security, such visualizations help ensure that models make decisions for the *right reasons*.

---

## How to Reproduce

1. Install requirements:

   ```bash
   pip install mtcnn opencv-python matplotlib torch torchvision pytorch-grad-cam
   ```
2. Place FairFace sample images into a folder (e.g., `images/`).
3. Run the notebook `explainable-deep-learning.ipynb`.

   * Results (heatmaps) will be saved in `results/` and `cropped/`.
