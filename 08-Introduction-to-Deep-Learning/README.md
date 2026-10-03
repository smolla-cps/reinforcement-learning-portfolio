# Introduction to Artificial Neural Networks and Deep Learning

A hands-on Python notebook explaining how artificial neural networks work, from a single neuron to a trained multilayer convolutional neural network (CNN). The notebook combines mathematical derivations, manual calculations, executable code, and visualizations.

**Notebook:** [ANN_and_Deep_Learning.ipynb](ANN_and_Deep_Learning.ipynb)

## Topics covered

| Topic | What the notebook demonstrates |
|---|---|
| Neural network foundations | AI, machine learning, deep learning, linear versus nonlinear models, features, neurons, perceptrons, and network layers |
| Forward propagation | Weighted sums, biases, dense layers, matrix operations, hidden layers, and XOR |
| Activation functions | Identity, sigmoid, tanh, ReLU, and Leaky ReLU; derivatives, saturation, vanishing gradients, and dying ReLU |
| Loss functions | Prediction versus target, mean squared error, binary cross-entropy, softmax, cross-entropy, and empirical mean loss |
| Backpropagation | Chain-rule derivatives and **manual calculations for every weight and bias** in a two-layer network, checked against PyTorch autograd and finite differences |
| Optimization | Gradient descent trajectories, learning-rate effects, full-batch gradient descent, one-sample SGD, mini-batches, momentum, Adagrad, RMSprop, and Adam |
| Generalization | Underfitting, overfitting, L2 regularization, dropout, batch normalization, and validation-based early stopping |
| Image processing | Image arrays, color channels, tensor shapes, convolution kernels, stride, padding, and filter arithmetic |
| CNNs | Manual convolution and pooling calculations, feature maps, CNN training, and architecture visualization |
| Other architectures | RNNs, LSTMs, GANs, radial basis function networks, self-organizing maps, LeNet-5, AlexNet, VGG, Inception, bottlenecks, and YOLO concepts |

## Worked examples

### Single-neuron and multilayer calculations

The notebook begins with individual weighted inputs and follows the calculations through activation, loss, derivatives, and parameter updates. It includes an exact XOR construction, a single-neuron gradient descent example, and a **two-input, two-hidden-neuron, one-output** network whose gradients are computed by hand and verified in PyTorch.

### Swimming competition classification

A small, **synthetically generated** dataset illustrates binary classification with swimming-practice and dryland-workout features. The example includes train/validation/test splits, input scaling, binary cross-entropy training, and learning curves. This is a teaching example, not a real-world performance study.

### Convolution and max pooling

A small numerical image and kernel are used to calculate convolution outputs by elementwise multiplication and summation. Additional exercises illustrate filter responses, stride/padding effects, convolution output dimensions, and max pooling.

### Multilayer CNN for digit classification

A supervised CNN is trained on the **8 × 8 handwritten-digit dataset bundled with scikit-learn**. Its components include convolution, batch normalization, ReLU, pooling, and fully connected layers. The notebook provides a generated architecture diagram, tensor-shape checks, training-loss and validation-accuracy plots, test evaluation, and learned feature-map visualizations.

The principal multilayer model follows this sequence:

```text
Grayscale image (1 × 8 × 8)
    → Conv2d(1, 8) + BatchNorm + ReLU
    → Conv2d(8, 16) + BatchNorm + ReLU
    → MaxPool2d(2)                       [16 × 4 × 4]
    → Conv2d(16, 32) + BatchNorm + ReLU
    → MaxPool2d(2)                       [32 × 2 × 2]
    → Flatten                            [128]
    → Linear(128, 64) + ReLU
    → Linear(64, 10)                     [class logits]
```

## Run the notebook

**Google Colab**

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload notebook** and choose `ANN_and_Deep_Learning.ipynb`.
3. Run the cells from top to bottom. The notebook uses the CPU and does not require a GPU.

**Local JupyterLab**

```bash
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

Install the dependencies and launch JupyterLab:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open `ANN_and_Deep_Learning.ipynb` in JupyterLab, select the Python kernel, and run the cells in order. **Python 3.10 or newer** is recommended.

## Data and reproducibility

- No external CSV files, downloaded datasets, model checkpoints, or API keys are required.
- The notebook generates its synthetic demonstration data locally and uses `sklearn.datasets.load_digits()` for digit classification.
- The random seed is set to `42` for the main experiments. Results may still vary slightly across library versions and execution environments.
- All training examples can run on CPU; execution time depends on the machine.
- The figures are generated by the notebook code, so they can be recreated after restarting the kernel and running all cells.

## Files

```text
.
├── ANN_and_Deep_Learning.ipynb   # Explanations, derivations, experiments, and figures
├── README.md                     # Project overview and execution instructions
└── requirements.txt              # Python dependencies
```
