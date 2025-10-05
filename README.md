# Clerum - Neural Network Library in Rust

**Clerum** is a fast and lightweight neural network engine written in Rust, designed with modularity and parallelism in mind. It is ideal for experimentation, custom pipelines, or as the core engine of your own AI systems.

## Features

- **Feedforward Neural Networks (FNN)**: Fully connected networks for classification and regression
- **Recurrent Neural Networks (RNN)**: Sequence modeling with support for sequence-to-sequence and sequence-to-vector tasks
- **Common activation functions**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, and more
- **Loss functions**: MSE, MAE, Huber, BCE, CrossEntropy
- **Optimizers**: SGD, Adam, RMSprop, Momentum
- **Parallel training** using Rayon
- **Model checkpointing** (.clr format)
- **Gradient clipping** for training stability

## Installation

Add Clerum to your `Cargo.toml` :

```toml
[dependencies]
clerum = { git = "https://github.com/FauzanFR/clerum.git", branch = "experiment" }
```

## Usage
### Feedforward Network (FNN) - Classification

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
### Feedforward Network (FNN) - Regression
```rust
use clerum::{activation::ActivationConfig, file::run_cler, helper::{rand_arr2d, set_global_seed}, 
             loss::LossConfig, optimizer::OptimizerConfig, FNN};
use ndarray::{array, Array2};

fn main() {
    let x: Array2<f32> = rand_arr2d(100, 2);
    
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
}
```
### Recurrent Neural Network (RNN) - Sequence Modeling
```rust
use clerum::{activation::ActivationConfig, loss::LossConfig, optimizer::OptimizerConfig, RNN, RNNTask};
use ndarray::{array, Array3};

fn main() {
    // Create time series data: (batch_size, sequence_length, features)
    let x = array![
        // Sample 1: Sine wave
        [[0.0], [0.5], [0.87], [1.0], [0.87], [0.5]],
        // Sample 2: Different phase
        [[1.0], [0.87], [0.5], [0.0], [-0.5], [-0.87]],
    ].into_dyn().into_dimensionality::<ndarray::Ix3>().unwrap();

    // Target: next value prediction (sequence-to-sequence)
    let y = array![
        [[0.5], [0.87], [1.0], [0.87], [0.5], [0.0]],
        [[0.87], [0.5], [0.0], [-0.5], [-0.87], [-1.0]],
    ];

    // Initialize RNN for sequence-to-sequence task
    let mut model = RNN::init(x, y, 6); // sequence_length = 6
    
    // Add RNN layers
    model.add_layer(1, 8, ActivationConfig::Tanh);  // Input: 1 feature, Hidden: 8 units
    model.add_layer(8, 4, ActivationConfig::Tanh);  // Second layer: 8 -> 4 units
    
    // Set output layer
    model.set_output_layer(1); // Output: 1 value per timestep

    // Train the RNN
    model.train(
        0.01,                   // Learning rate
        1.0,                    // Gradient clipping norm
        100,                    // Epochs
        LossConfig::MSE,        // Loss function
        false,                  // Pretraining
        0.0,                    // Pretrain ratio
        0,                      // Pretrain epochs
        0,                      // Parallel threads (0 = auto)
        "rnn_model.clr",        // Output path
        OptimizerConfig::Adam(0.9, 0.999) // Optimizer
    );
}
```
### Feedforward Network (FNN) - Regression
```rust
use clerum::{activation::ActivationConfig, loss::LossConfig, optimizer::OptimizerConfig, RNN, RNNTask};
use ndarray::{array, Array3};

fn main() {
    // Create time series data: (batch_size, sequence_length, features)
    let x = array![
        // Sample 1: Sine wave
        [[0.0], [0.5], [0.87], [1.0], [0.87], [0.5]],
        // Sample 2: Different phase
        [[1.0], [0.87], [0.5], [0.0], [-0.5], [-0.87]],
    ].into_dyn().into_dimensionality::<ndarray::Ix3>().unwrap();

    // Target: next value prediction (sequence-to-sequence)
    let y = array![
        [[0.5], [0.87], [1.0], [0.87], [0.5], [0.0]],
        [[0.87], [0.5], [0.0], [-0.5], [-0.87], [-1.0]],
    ];

    // Initialize RNN for sequence-to-sequence task
    let mut model = RNN::init(x, y, 6); // sequence_length = 6
    
    // Add RNN layers
    model.add_layer(1, 8, ActivationConfig::Tanh);  // Input: 1 feature, Hidden: 8 units
    model.add_layer(8, 4, ActivationConfig::Tanh);  // Second layer: 8 -> 4 units
    
    // Set output layer
    model.set_output_layer(1); // Output: 1 value per timestep

    // Train the RNN
    model.train(
        0.01,                   // Learning rate
        1.0,                    // Gradient clipping norm
        100,                    // Epochs
        LossConfig::MSE,        // Loss function
        false,                  // Pretraining
        0.0,                    // Pretrain ratio
        0,                      // Pretrain epochs
        0,                      // Parallel threads (0 = auto)
        "rnn_model.clr",        // Output path
        OptimizerConfig::Adam(0.9, 0.999) // Optimizer
    );
}
```

## API Overview

### Network Types
- **FNN**: Feedforward Neural Network for static data
- **RNN**: Recurrent Neural Network for sequence data

### RNN Task Types
- **Sequence-to-Sequence**: Output at each timestep (e.g., time series prediction)
- **Sequence-to-Vector**: Single output at end of sequence (e.g., sequence classification)

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

### FNN Methods
- `add_layer(input_dim, output_dim, activation)`: Add a layer to the network
- `add_labels(labels)`: Add class labels for classification
- `train(lr, epochs, loss, pretrain, pretrain_ratio, pretrain_epochs, threads, path, optimizer)`: Train the model
- `from_checkpoint(path)`: Load pre-trained weights

### RNN Methods
- `add_layer(input_dim, hidden_dim, activation)`: Add an RNN layer
- `set_output_layer(output_dim)`: Set the output layer dimensions
- `add_labels(labels)`: Add class labels for classification
- `train(lr, max_norm, epochs, loss, pretrain, pretrain_ratio, pretrain_epochs, threads, path, optimizer)`: Train the model

### Utility Functions
- `run_cler(model_path, input, confidence_threshold)`: Make predictions with FNN models
- `set_global_seed(seed)`: Set random seed for reproducibility

## Project Structure
```
src/
├── main.rs                 # Example usage
├── lib.rs                  # Library entry point
└── clerum/                 # Core library
    ├── mod.rs              # Module declarations (FNN, RNN)
    ├── activation.rs       # Activation functions
    ├── file.rs             # Model serialization
    ├── helper.rs           # Utility functions
    ├── loss.rs             # Loss functions
    ├── optimizer.rs        # Optimization algorithms
```
## Experimental Features
⚠️ Note: The RNN implementation is currently in the `experiment` branch and should be considered experimental. It includes:

- Multi-layer RNN support
- Gradient clipping for training stability
- Sequence-to-sequence and sequence-to-vector tasks
- Basic time series prediction capabilities

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes with descriptive messages
4. Push your branch and create a pull request

For experimental features like RNN, please use the `experiment` branch.

## License

Licensed under the [Apache License 2.0](./LICENSE).  
You are free to use, modify, and distribute this project, provided that proper attribution is given.
