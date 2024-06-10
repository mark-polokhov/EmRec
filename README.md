# Emotion Recognition Project

This repository contains the code and resources for the Emotion Recognition Project. The project aims to develop a robust and reliable solution for recognizing human emotions from video data, focusing on facial expressions.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Experiments](#experiments)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Emotion recognition encompasses diverse aspects such as facial expression analysis, vocal intonation, and body language interpretation. This project focuses on facial components using CNNs and Transformers to classify emotional states from video frames. The primary goal is to provide a user-friendly web application for emotion recognition from uploaded video files.

## Features

- **Emotion Classification**: Classifies emotions from video frames.
- **Web Application**: User-friendly interface for uploading videos and receiving frame-by-frame emotion descriptions.
- **Model Architectures**: Implements and compares various architectures including CNNs and VisionTransformer.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/emotion-recognition.git
    cd emotion-recognition
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the web application**:
    ```bash
    python main.py
    ```
    or
    ```bash
    python3 main.py
    ```

3. **Upload a video file**: Use the web interface to upload your video file.

4. **Get descriptions**: The application will process the video and provide frame-by-frame emotion descriptions.

## Model Architectures

The project explores various model architectures for emotion recognition:

- **CNN (Convolutional Neural Network)**
- **VisionTransformer**

## Experiments

The experiments conducted in this project include:

- Comparison of model performance on the AffectNet-short dataset.
- Evaluation of DINO segmentation for improving recognition accuracy.
- Analysis of VisionTransformer architectures.

## Results

The experimental results demonstrate the performance of different models on the AffectNet dataset. Detailed results and comparisons can be found in the WandB workspace.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## Contact

For any inquiries or feedback, please contact us at [myupolokhov@edu.hse.ru](mailto:myupolokhov@edu.hse.ru).

