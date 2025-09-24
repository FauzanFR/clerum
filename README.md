# Clerum - Neural Network Library in Rust

**Clerum** is a fast and lightweight neural network engine written in Rust, designed with modularity and parallelism in mind. It is ideal for experimentation, custom pipelines, or as the core engine of your own AI systems.

## Features

- Modular feedforward neural networks (FNN)
- Common activation functions: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- Loss functions: MSE, CrossEntropy
- Optimizers: SGD, Adam, RMSprop, Momentum
- Optional pretraining
- Parallel training using Rayon
- Model checkpointing (.clr format)

## Installation

Add Clerum to your `Cargo.toml` :

```toml
[dependencies]
clerum = { git = "https://github.com/FauzanFR/clerum.git" }
```

## Usage
### Classification Example

```rust
use clerum::{activation::ActivationConfig, file::run_cler, helper::{rand_arr2d, set_global_seed}, 
             loss::LossConfig, optimizer::OptimizerConfig, FNN};
use ndarray::{array, Array2};

fn main() {
    // Generate random data: 100 samples, 2 features
    let x: Array2<f32> = rand_arr2d(100, 2);
    
    // Input for prediction
    let input = array![[0.2, 0.0]];
    
    // Class labels
    let labels = vec![3, 2];

    // Create one-hot encoded targets
    let mut y_array = Array2::zeros((x.nrows(), labels.len()));
    for (i, row) in x.rows().into_iter().enumerate() {
        let label_val = if row[0] + row[1] < 1.0 { 2 } else { 3 };
        let idx = labels.iter().position(|&x| x == label_val).unwrap();
        y_array[[i, idx]] = 1.0;
    }
    
    // Initialize model
    let mut model = FNN::init(x, y_array);
    
    // Set seed
    set_global_seed(42);
    
    // Add layers
    model.add_layer(2, 16, ActivationConfig::LeakyReLU(0.01));
    model.add_layer(16, 8, ActivationConfig::LeakyReLU(0.01));
    model.add_layer(8, 2, ActivationConfig::Softmax);
    
    // Add class labels
    model.add_labels(labels);

    // Train model
    model.train(
        0.01,                   // Learning rate
        60,                     // Epochs
        LossConfig::CrossEntropy, // Loss function
        true,                   // Use pretraining
        0.1,                    // Pretrain data ratio
        5,                      // Pretrain epochs
        4,                      // Parallel threads
        "model.clr",            // Model file
        OptimizerConfig::Adam(0.9, 0.999) // Optimizer
    );

    // Make prediction
    match run_cler("model.clr", input, 0.50) {
        Some((idx, confidence)) => {
            println!("Prediction: Class {} | Confidence: {:.2}%", idx, confidence * 100.0)
        },
        None => println!("Model is not confident enough"),
    }
}
```
### Regression Example
```rust
// ... (initialization code similar to classification)

// Create regression targets
let y = x.map_axis(ndarray::Axis(1), |row| row[0] + row[1]).insert_axis(ndarray::Axis(1));

// Set seed
set_global_seed(42);

let mut model = FNN::init(x, y);
model.add_layer(2, 16, ActivationConfig::LeakyReLU(0.01));
model.add_layer(16, 8, ActivationConfig::LeakyReLU(0.01));
model.add_layer(8, 4, ActivationConfig::LeakyReLU(0.01));
model.add_layer(4, 1, ActivationConfig::Linear);

model.train(
    0.001,
    100,
    LossConfig::Huber(1.0),
    false,
    0.0,
    0,
    4,
    "reg_model.clr",
    OptimizerConfig::Adam(0.9, 0.999)
);

// Make prediction
let input = array![[0.5, 0.3]];
match run_cler("reg_model.clr", input, 0.0) {
    Some((idx, value)) => println!("Predicted value: {:.4}", value),
    None => println!("No prediction"),
}
```

## API Overview
### Activation Functions

- `ReLU` `Sigmoid`, `Tanh`, `LeakyReLU(alpha)`, `ELU(alpha)`, `Softplus`, `Linear`, `Softmax`
- Derivatives available for all activation functions

### Loss Functions
- `MSE`, `MAE`, `Huber(delta)`, `BCE`, `WeightedBCE(weight)`, `CrossEntropy`
- Derivatives available for all loss functions

### Optimizers
- `SGD`
- `Momentum(gamma)`
- `RMSprop(gamma)`
- `Adam(beta1, beta2)`

### Model Methods
- `add_layer(input_dim, output_dim, activation)`: Add a layer to the network
- `add_labels(labels)`: Add class labels for classification
- `train(lr, epochs, loss, pretrain, pretrain_ratio, pretrain_epochs, threads, path, optimizer)`: Train the model
- `run_cler(model_path, input, confidence_threshold)`: Make predictions

### Manual Save/Load

Clerum allows you to manually save and load model states using checkpoint files (.clr) for greater control or custom workflows.

#### Save a model manually:

```rust
use clerum::file::save_checkpoint;

let checkpoint = model.checkpoint(); // snapshot of current model
save_checkpoint("model.clr", &checkpoint).unwrap();

```
#### load model:
```rust
use clerum::file::load_checkpoint;

let checkpoint = load_checkpoint("model.clr").unwrap();
```
#### Load a model manually into an existing FNN:
```rust
use clerum::FNN;
// retrain the model
let mut model = FNN::init(dummy_x, dummy_y);// data to be used for retraining
model.from_checkpoint("model.clr"); // load model

// Re-train with different optimizer or loss
model.train(
    0.001,
    20,
    LossConfig::MSE,  // different loss
    false,
    0.0,
    0,
    4,
    "model_retrained.clr",
    OptimizerConfig::SGD(0.9) // different optimizer
);
```
Note: `from_checkpoint(path)` mutates an existing model with the saved weights, biases, activation functions, loss function, and labels.

Use `FNN::init(...)` first before calling it, especially if you want to continue training.


### Checkpoint Contents

The saved checkpoint contains the following internal state

```rust
pub struct Checkpoint {
    epoch: usize,                   // Last trained epoch
    loss: f32,                      // Final loss value
    weights: PackedTensor2DStorage, // All weights for each layer
    biases: PackedTensor1DStorage,  // All biases for each layer
    activation_id: Vec<usize>,      // Activation function IDs per layer
    data_act: Vec<f32>,             // Extra activation config (e.g., LeakyReLU alpha)
    loss_id: usize,                 // Loss function ID
    data_loss: f32,                 // Extra loss config (e.g., Huber delta)
    labels: Vec<i32>,               // Label mapping (classification only)
}
```
You can freely change the loss function and optimizer during retraining.
Just ensure the model structure and label mapping remain consistent.
## Project Structure
```
src/
├── main.rs                 # Example usage
├── lib.rs                  # Library entry point
└── clerum/                 # Core library
    ├── mod.rs              # Module declarations
    ├── activation.rs       # Activation functions
    ├── file.rs             # Model serialization
    ├── helper.rs           # Utility functions
    ├── loss.rs             # Loss functions
    └── optimizer.rs        # Optimization algorithms
```
## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes with descriptive messages
4. Push your branch and create a pull request

## License

Licensed under the [Apache License 2.0](./LICENSE).  
You are free to use, modify, and distribute this project, provided that proper attribution is given.
