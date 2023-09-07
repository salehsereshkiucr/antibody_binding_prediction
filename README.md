# Predicting Antibody Binders using Deep Learning

This repository contains Python code for deeper analysis of the results of the paper "Predicting antibody binders and generating synthetic antibodies using deep learning".

We implemented models using CNN (Convolutional Neural Networks), Random Forest, and Decision Trees to predict the antibody binding for CTLA-4 and PD-1.

## Results

Here are some results from our implementation:

![Result Image 1](./results/PD-1-ROC-regenerate.png)
![Result Image 2](./results/PD-1-probability-regenerate.png)
![Result Image 1](./results/PD-1-regenerate.png)
![Result Image 1](./results/CTLA-4-ROC-regenerate.png)
![Result Image 2](./results/CTLA-4-binder_probability-regenerate.png)
![Result Image 1](./results/CTLA-4-regenerating.png)

Also, we did some feature analysis of the input amino acid sequence. Here is a plot of the amino acid importance heatmap. This result is generated based on the feature analysis of a Decision Tree:

![Result Image 1](./results/PD-1_feature_importance.png)
![Result Image 1](./results/CTLA-4_feature_importance.png)
