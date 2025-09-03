# Cross-Model Watermarking via Discriminative Samples for Secure Authentication

![ACM MM 2025](https://img.shields.io/badge/ACM-MM%202025-blue)  
**Paper accepted at ACM Multimedia 2025 (Dublin, Ireland)**

## Overview

This repository contains the official PyTorch implementation for our ACM MM 2025 paper:

**Cross-Model Watermarking via Discriminative Samples for Secure Authentication**  
*Juan Zhao, Yudao Sun\*, Zhihai Yang, Cai Xu, Hongji Chen, Fan Zhang, Jianxin Li*  
[Paper PDF](link_to_camera_ready_or_arxiv)

---

## Abstract

Deep neural networks on cloud platforms face increasing security threats, especially as AI services often deploy diverse models for the same task. Existing watermarking methods struggle to distinguish benign modifications from malicious attacks in cross-model scenarios.  
We propose a **non-intrusive cross-model watermarking method** that generates discriminative samples as universal keys, enabling robust and transferable authentication **without modifying model parameters or architectures**. A novel margin enhancement loss amplifies the confidence gap between benign and malicious behaviors, ensuring high transferability and strong discriminability across models.

**Key highlights:**
- **Non-intrusive:** No need to change model parameters or structure.  
- **Highly transferable:** Works across different architectures for the same task.  

For more details, please refer to the full paper.

---

## Installation

**Required environment:**  
- Python ≥ 3.8  
- PyTorch == 2.4.1  
- torchvision == 0.19.1  
- numpy == 1.24.4  
- tqdm == 4.67.1  

Install dependencies via:
```bash
pip3 install -r requirements.txt
```

---

## Project Structure

```
.
├── attacks/                   # Attack scripts and settings
├── models/                    # Pre-trained models or model definitions
├── nets/                      # Neural network architecture definitions
├── utils/                     # Utility functions
├── attack_main.py             # Attack launching script
├── authentication.py          # Verification and authentication procedure
├── distinguishable_method.py  # Core implementation of watermarking method
├── requirements.txt           # Environment dependencies
├── README.md                  # Project documentation
```

---

## Workflow: Discriminative Sample Generation, Attack, and Authentication

This project provides a complete workflow for generating discriminative samples, attacking pre-trained models, and verifying the effects.

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Download pre-trained models

Download `resnet50.pth` and `vgg19.pth` into the `models/` directory before running experiments.

### 3. Generate discriminative samples

Use pre-trained models to generate discriminative samples. The argument `--exp_tag` is a unique identifier for each experiment (replace `202508151725` with your custom tag).

* Generate samples using ResNet-50:
```bash
python3 distinguishable_method.py --model resnet50 --exp_tag 202508151725
```

* Generate samples using VGG-19:
```bash
python3 distinguishable_method.py --model vgg19 --exp_tag 202508151725
```

### 4. Attack pre-trained models

We simulate fine-tuning (`fine_tune`) and quantization (`quantization`) attacks.

* Attacking VGG-19:
```bash
python3 attack_main.py --model vgg19 --exp_tag 202508151725 --attack fine_tune
python3 attack_main.py --model vgg19 --exp_tag 202508151725 --attack quantization
```

* Attacking ResNet-50:
```bash
python3 attack_main.py --model resnet50 --exp_tag 202508151725 --attack fine_tune
python3 attack_main.py --model resnet50 --exp_tag 202508151725 --attack quantization
```

### 5. Authentication

Verify the attack effect using discriminative samples generated from one model (`surrogate_model`) against another (`target_model`).

* Using ResNet-50 samples to verify VGG-19:
```bash
python3 authentication.py --surrogate_model resnet50 --target_model vgg19 --attack fine_tune --exp_tag 202508151725
python3 authentication.py --surrogate_model resnet50 --target_model vgg19 --attack quantization --exp_tag 202508151725
```

* Using VGG-19 samples to verify ResNet-50:
```bash
python3 authentication.py --surrogate_model vgg19 --target_model resnet50 --attack quantization --exp_tag 202508151725
python3 authentication.py --surrogate_model vgg19 --target_model resnet50 --attack fine_tune --exp_tag 202508151725
```

**Note:** All results will be saved in a directory named after the `--exp_tag`. Authentication results are stored in `authentication.json`.

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{zhao2025cross,
  title={Cross-Model Watermarking via Discriminative Samples for Secure Authentication},
  author={Zhao, Juan and Sun, Yudao and Yang, Zhihai and Xu, Cai and Chen, Hongji and Zhang, Fan and Li, Jianxin},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year={2025},
  organization={ACM},
  doi={10.1145/3746027.3755177}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

For questions or collaborations, please contact the corresponding author:  
Yudao Sun (sunyd@pcl.ac.cn)

---

## Acknowledgment

This work was supported in part by the National Key R&D Program of China, Shaanxi Province Science Foundation Fund, and other grants (see paper for details).
