## Overview
This repository contains my work for the Cynaptics Club inductions at IIT Indore. The tasks consist of three main sub-tasks, which I have worked on and will describe below. Each task had specific requirements and evaluations, and I have detailed my approach to solving them.

## Tasks

### Compulsory Sub-Task 1: AI vs Real Image Classifier (40% weightage)
In this task, participants were free to use any algorithm of their choice. The goal was to build a model that could distinguish between real and AI-generated images. This task included two contests on Kaggle.
1. **Starting Point**:
   I began by using the baseline code provided by the Cynaptics Club after returning to campus from my vacations. My first step was to go through the code thoroughly, understanding it line by line by researching each part online.
2. **Understanding the Model**:
   While I managed to understand most of the code, I struggled with understanding the CNN (Convolutional Neural Network) model part. I plan to revisit this section in the future to deepen my understanding.
3. **Code Modifications**:
   I added a code segment to output the model’s results into a CSV file. During testing, I encountered an issue with certain images that caused errors. To handle this, I modified the code to skip these images and manually labeled them as "Real" in the CSV file.
4. **Model Training**:
   After completing these modifications, I achieved an initial accuracy of 0.96. However, I encountered fluctuations in accuracy during subsequent trials, with values dipping to 88% and even 64%. I then implemented a strategy to save the model's weights and iterate over them, which improved the accuracy to 0.975. Eventually, I achieved a perfect accuracy of 1.0.
5. **Second Contest**:
   I applied the same logic to the second contest, and as of now, the maximum accuracy I have achieved is 0.55.

### Compulsory Sub-Task 2: Implement a GAN (50% weightage)
The task required participants to implement a Generative Adversarial Network (GAN) from scratch, with a baseline script provided for reference. The GAN should be able to generate good images, and bonus points were given if it could be used for an image classification task as well.
1. **Initial Attempt**:
   I started with the baseline script provided by the Cynaptics Club, but it was not able to handle my data effectively. As a result, I turned to online resources and implemented a basic GAN architecture from scratch.
2. **Dataset and Training**:
   I chose a simple dataset of leaves to train the model. I focused mainly on hyperparameter tuning, experimenting with batch sizes and the number of epochs to improve the model’s performance.
3. **Outcome**:
   The GAN was able to generate images of leaves, but I did not yet achieve high-quality results. Further tuning and model refinement are necessary to improve the output.

### Bonus Sub-Task: Train Your GAN on a Theme-Based Dataset (10% weightage)
In this bonus task, participants were encouraged to train their GAN on a theme-based dataset, with a Christmas theme as a starting point. Participants were free to experiment with other themes as well. Unfortunately, due to time constraints, I was unable to complete this bonus task. As a result, I have skipped this task for now.

## Reference
(https://github.com/CynapticsAI/InductionDec2024/tree/main).
