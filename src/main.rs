use clerum::{activation::ActivationConfig, file::run_cler, helper::rand_arr2d, loss::LossConfig, optimizer::OptimizerConfig, FNN};
use ndarray::{array, Array2};

fn main() {
    // 1. Data Preparation
    // -----------------
    // Generate random data: 10 samples, 2 features
    let x: Array2<f32> = rand_arr2d(10, 2);
    
    // Input for prediction after training
    let input = array![[0.2, 0.0]];
    
   // Class labels: 2 and 3
    let labels = vec![3, 2];

    // Create a one-hot encoded target
    let mut y_array = Array2::zeros((x.nrows(), labels.len()));
    for (i, row) in x.rows().into_iter().enumerate() {
        // Simple rule: if feature1 + feature2 < 1 then class 2, else class 3
        let label_val = if row[0] + row[1] < 1.0 { 2 } else { 3 };
        let idx = labels.iter().position(|&x| x == label_val).unwrap();
        y_array[[i, idx]] = 1.0;
    }
    
    // 2. Model Initialization
    // ---------------------
    let mut model = FNN::init(x, y_array);
    
    // Add a network layer:
    model.add_layer(2, 16, ActivationConfig::LeakyReLU(0.01));      // Input: 2 features
    model.add_layer(16, 8, ActivationConfig::LeakyReLU(0.01));      // Hidden layer
    model.add_layer(8, 2, ActivationConfig::Softmax);               // Output: 2 classes
    
    // Add class label (MANDATORY for classification)
    model.add_labels(labels);

    // 3. Training Model
    // -----------------
    model.train(
        0.01,                                      // Learning rate
        30,                                     // Epochs
        LossConfig::CrossEntropy,           // Loss function
        true,                                // Use pretraining?
        0.1,                           // Pretrain data ratio
        5,                            // Epoch pretrain
        4,                           // Thread paralel
        "model.clr",                             // Model file name
        OptimizerConfig::Adam(0.9, 0.999)   // Optimizer Adam
    );

    // 4. Prediction
    // ------------
    match run_cler("model.clr", input, 0.75) {
        Some((idx, confidence)) => {
            println!("Prediction: Class {} | Confidence: {:.2}%", idx, confidence * 100.0)
        },
        None => println!("The model is not confident in its predictions."),
    }
}