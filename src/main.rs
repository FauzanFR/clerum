use std::f32::consts::PI;

use clerum::{FNN::{self, BatchConfig, PretrainConfig}, RNN, activation::ActivationConfig, file::{Input_X, run_cler}, helper::{rand_arr2d, rand_arr3d, set_global_seed}, loss::LossConfig, optimizer::OptimizerConfig};
use ndarray::{Array2, Array3, ArrayD, array, s};

// fn main() {
//     // 1. Data Preparation
//     // -----------------
//     // Generate random data: 100 samples, 2 features
//     let x: Array2<f32> = rand_arr2d(100, 2);
    
//     // Input for prediction after training
//     let input = array![[0.2, 0.0]];
    
//    // Class labels: 2 and 3
//     let labels = vec![2, 3];

//     // Create a one-hot encoded target
//     let mut y_array = Array2::zeros((x.nrows(), labels.len()));
//     for (i, row) in x.rows().into_iter().enumerate() {
//         // Simple rule: if feature1 + feature2 < 1 then class 2, else class 3
//         let label_val = if row[0] + row[1] < 1.0 { 2 } else { 3 };
//         let idx = labels.iter().position(|&x| x == label_val).unwrap();
//         y_array[[i, idx]] = 1.0;
//     }
    
//     // 2. Model Initialization
//     // ---------------------
//     let mut model = FNN::FNN::init(x, y_array);
    
//     // Set seed
//     set_global_seed(42);

//     // Add a network layer:
//     model.add_layer(2, 16, ActivationConfig::LeakyReLU(0.01));      // Input: 2 features
//     model.add_layer(16, 8, ActivationConfig::LeakyReLU(0.01));      // Hidden layer
//     model.add_layer(8, 2, ActivationConfig::Softmax);               // Output: 2 classes
    
//     // Add class label (MANDATORY for classification)
//     model.add_labels(labels);

//     // 3. Training Model
//     // -----------------
//     model.train(
//         0.01,                                      // Learning rate
//         60,                                     // Epochs
//         LossConfig::CrossEntropy,           // Loss function
//         PretrainConfig::Partial { data_ratio: 0.1, epochs: 6 },
//         BatchConfig::Sequential,
//         "model.clr",                             // Model file name
//         OptimizerConfig::Adam(0.9, 0.999)   // Optimizer Adam
//     );

//     // 4. Prediction
//     // ------------
//     match run_cler("model.clr", input, 0.50) {
//         Some((idx, confidence)) => {
//             println!("Prediction: Class {} | Confidence: {:.2}%", idx, confidence * 100.0)
//         },
//         None => println!("The model is not confident in its predictions."),
//     }
// }



fn main() {
    // Parameter utama
    let samples = 100;
    let steps = 20;
    let features = 1;

    // Tempat data mentah
    let mut data = Vec::new();

    for s in 0..samples {
        for t in 0..steps {
            let val = match s {
                0 => (t as f32 * PI / 6.0).sin(),
                1 => (t as f32 * PI / 6.0 + PI / 2.0).sin(),
                2 => t as f32 / (steps as f32 - 1.0),
                _ => 0.0,
            };
            data.push(val);
        }
    }

    let mut ydata = Vec::new();

    for s in 0..samples {
        for t in 0..steps {
            let val = match s {
                0 => ((t + 1) as f32 * PI / 6.0).sin(),
                1 => ((t + 1) as f32 * PI / 6.0 + PI / 2.0).sin(),
                2 => ((t + 1).min(steps - 1)) as f32 / (steps as f32 - 1.0),
                _ => 0.0,
            };
            ydata.push(val);
        }
    }

    let y = Array3::from_shape_vec((samples, steps, features), ydata).unwrap();
    let X = Array3::from_shape_vec((samples, steps, features), data).unwrap();

    // let mut model = RNN::RNN::init(X, y, None);
    // model.add_layer(1, 4, ActivationConfig::Tanh);
    // model.add_layer(4, 7, ActivationConfig::ReLU);
    // model.add_layer(7, 5, ActivationConfig::LeakyReLU(0.1));
    // model.add_layer(5, 5, ActivationConfig::Softmax);
    // model.train(0.01, 10.0, 40, LossConfig::Huber(0.1), true, 0.1, 5, "test.clr", OptimizerConfig::Adam(0.9, 0.999));

    run_cler("test.clr",Input_X::Array3(X), 0.70);

    println!("y_true {:?}", y)
}