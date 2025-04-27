# ASLert: Emergency American Sign Language (ASL) Interpreter
## Table of Contents
1. [Who are we?](#who-are-we)
2. [What is this?](#what-is-this)
3. [Why is this \[important\]?](#why-is-this-important)
4. [How does this work?](#how-does-this-work)
5. [Test it yourself](#test-it-yourself)

## Who are we?
We are a team of students from Georgia State University under [Dr. Esra Akbas]()' Spring 2025 Honors Machine Learning course. Our team includes:

- [Niko Avradopoulos](https://www.linkedin.com/in/nikoavra/)
- [Rudra Patel](https://www.linkedin.com/in/rudra-patel-6a9080295/)
- [Aditya Sharma](https://www.linkedin.com/in/aditya-sharma-32a379291/)

## What is this?
> _This is a project for our Honors Machine Learning course. Consequently, updates to this repository musn't be expected, but you may still contact our team_ ___here?___ _for any inquiries._

ASLert is an emergency American Sign Language (ASL) interpreter developed to offer real-time, automatic ASL translation during emergencies. The system leverages neural-network-powered models to translate critical emergency phrases and gestures into ASL, ensuring swift and accurate communication during urgent situations.

## Why is this [important]?
Effective communication during emergencies is often difficult for the Deaf community, as they may face barriers in understanding or relaying critical information. ASLert addresses this challenge by providing a tool for ASL interpretation. Our hope was to ensure that emergency messages are communicated clearly and promptly, thus improving safety and accessibility in urgent situations.

## How does this work?
The ASLert system processes video input data to recognize and interpret emergency ASL gestures. We tested three different neural networks to analyze and understand these gestures: RNN, LSTM, & 3D-CNN. We've structured our repository logically, with an emphasis on ease-of-access.

- __Pre-Processing__: Video inputs are processed using MediaPipe's Hand Landmarker (see their docs [here](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)) and otherwise filtered & normalized to extract key features essential for ASL recognition, including motion patterns and hand shapes.
- __Models__: Each model follows a similar simple structure: Input -> NN blocks -> Classification
- __Activation Functions__: Each neural network uses common activation functions like ReLU to introduce non-linearity and improve the network’s learning ability.
- __Regularization__: Dropout and L2 regularization techniques prevent overfitting, ensuring the model generalizes well across different inputs.
- __Implementation__: The system was tested using both automated and manual methods to assess accuracy and robustness across a variety of input data.
- __Training__: The networks are trained with typical ML methodologies (gradient descent, backpropagation) to optimize the model's performance and accuracy.

## Test it yourself
To test ASLert on your own, we've outlined a simple process:

- __Data__: The `data` directory holds... you guessed it: data. The `.labels.json` file holds pertinent metadata of each file.
  - _Adding your own data?_ Sure! Video format shouldn't matter - simply add your video(s) and label it/them (within `.labels.json`).
- __Models__: We've attached pre-compiled models in `src/models`.
  - _Compiling your own model?_ No problem! Take a look at our RNN, LSTM, and 3D-CNN source code for reference. If you're compiling your own model, you likely know enough to get by. Again, ASLert's goal is to provide a _mobile_ solution, so these models are pretty simple.
- __Testing__: Our compilation scripts include built-in testing.
  - _What about real-time testing?_ See the `demo` directory for live demos. `live_demo_mp.py` is currently set to demo our latest, most accurate RNN & is configured for our models that use MediaPipe's Hand Landmarker (RNN, LSTM). You'll have to set the local `MODEL_PATH` variable & configure the system environment path (near the top of the script) to get it working on your machine.
