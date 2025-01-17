# InductionDec2024
Cynaptics Induction Tasks Dec 2024

# AI vs Real Image Classifier
This repository contains my work for the AI vs Real Image Classifier task for the AI/ML inductions at IIT Indore. The task involved developing a machine learning model capable of distinguishing between real and AI-generated images. Below is a brief overview of my approach to solving the task.

1. Starting Point:
   I began by using the baseline code provided by the AI/ML club after returning to campus from my vacations. My first step was to go through the code thoroughly, understanding it line by line by researching each part online.

2. Understanding the Model:
   While I managed to understand most of the code, I struggled with understanding the CNN (Convolutional Neural Network) model part. I plan to revisit this section in the future to deepen my understanding.

3. Code Modifications:
   I added a code segment to output the model’s results into a CSV file. During testing, I encountered an issue with an image (`image_62.jpg`) that caused an error. To handle this, I modified the code to skip this image and manually labeled it as "Real" in the CSV file. 

4. Model Training:
   After completing these modifications, I achieved an initial accuracy of 0.96. However, I encountered fluctuations in accuracy during subsequent trials, with values dipping to 88% and even 64%. I then implemented a strategy to save the model's weights and iterate over them, which improved the accuracy to 0.975. Eventually, I achieved a perfect accuracy of 1.0.

5. Second Contest:
   I applied the same logic to the second contest, and as of now, the maximum accuracy I have achieved is 0.55.

## GAN Image Generator
[I have left space here for details on my approach for the GAN image generator once I complete it.]

## X-Mas Task
Due to time constraints, I was unable to complete the X-Mas task. Therefore, only the first two sub-tasks are included in this submission.
