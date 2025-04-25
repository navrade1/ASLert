ASLert: Emergency American Sign Language (ASL) Interpreter
1. Who We Are
We are a team of students from the Georgia State University Machine Learning class, working together to develop an innovative solution for real-time ASL interpretation in emergency situations. Our team includes:

Niko Avradopoulos

Rudra Patel

Aditya Sharma

For inquiries or more information, feel free to reach us at:

LinkedIn: [Insert LinkedIn URLs]

Student Emails: adibasel04@gmail.com, nikoavradopoulos@gmail.com

2. What This Is
ASLert is an emergency American Sign Language (ASL) interpreter developed to offer real-time, automatic ASL translation during emergencies. The system leverages video input and AI-powered models to translate critical emergency phrases and gestures into ASL, ensuring swift and accurate communication during urgent situations.

3. Why This Is Important
Effective communication during emergencies is often difficult for the deaf and hard-of-hearing community, as they may face barriers in understanding critical information. ASLert addresses this challenge by providing a reliable tool for ASL interpretation, ensuring that emergency messages are communicated clearly and promptly, thus improving safety and accessibility in urgent situations.

4. How This Works
The ASLert system processes video input data to recognize and interpret emergency ASL gestures. The model utilizes three neural networks to analyze and understand these gestures:

Activation Functions: Each neural network uses common activation functions like ReLU to introduce non-linearity and improve the network’s learning ability.

Regularization: Dropout and L2 regularization techniques are employed to prevent overfitting, ensuring the model generalizes well across different inputs.

Implementation: The system was tested using both automated and manual methods to assess accuracy and robustness across a variety of input data.

Model Details: The models are trained on a dataset of emergency ASL phrases paired with corresponding video data, ensuring the system learns to identify a wide range of relevant gestures.

Preprocessing: Video inputs are preprocessed to extract key features essential for ASL recognition, including motion patterns and hand shapes.

Training: The networks are trained using gradient descent and backpropagation to optimize the model's performance and accuracy.

5. How to Test It
To test the system with your own data, follow these steps:

Import Data: Load your video files or datasets into the system using the provided format.

Compile Models: Use the pre-trained models or follow the instructions in the demo directory to train your own models.

Run Testing: Input your data into the demo directory and run the models to see the translated ASL gestures.

The demo directory contains scripts for testing, and you can simply run them to try out the system with sample data.

6. Conclusion
ASLert provides a cutting-edge solution for translating emergency phrases into ASL using AI models. The integration of multiple neural networks ensures high accuracy and efficiency in recognizing video data. This system has the potential to significantly improve communication during emergencies for the deaf and hard-of-hearing community, offering a safer, more inclusive response to critical situations.

