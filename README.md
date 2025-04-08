# GSoC 2025 Evaluation Tasks: CMS and E2E Deep Learning Projects @ ML4SCI

This repository contains solutions for the Google Summer of Code (GSoC) 2025 evaluation tasks under the **ML4SCI** umbrella organization. The tasks evaluate your skills for the **End-to-End Deep Learning (E2E)** and **CMS** projects.

## Tasks Overview

1. **Common Task: Electron/Photon Classification**
   - **Dataset**: [Photon](https://cernbox.cern.ch/index.php/s/AtBT8y4MiQYFcgc) | [Electron](https://cernbox.cern.ch/index.php/s/FbXw3V4XNyYB3oA)
   - **Goal**: Classify electrons and photons using a ResNet-15-like model. Train on 80% of the data and evaluate on 20%. Use **PyTorch** or **Keras**.
   
2. **Specific Task: Event Classification with Masked Transformer Autoencoders**
   - **Dataset**: [HIGGS Dataset](https://archive.ics.uci.edu/dataset/280/higgs) (First 1.1 million events)
   - **Goal**: Train a Transformer Autoencoder on the dataset, using the first 21 features and 1.1 million events. Evaluate the classifier with **ROC-AUC** score.
