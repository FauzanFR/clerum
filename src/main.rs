use clerum::{activation::{self, ActivationConfig}, file::run_cler, helper::{rand_arr2d, rand_arr3d, set_global_seed}, loss::LossConfig, optimizer::OptimizerConfig, FNN, RNN};
use ndarray::{array, Array2, Array3, ArrayD};

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
//     let mut model = FNN::init(x, y_array);
    
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
//         true,                                // Use pretraining?
//         0.1,                           // Pretrain data ratio
//         5,                            // Epoch pretrain
//         4,                           // Thread paralel
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

fn create_timeseries_data() -> (Array3<f32>, Array3<f32>) {
    let batch_size = 3;
    let seq_len = 6;
    let input_dim = 1;
    let output_dim = 1;
    
    // X: Sine waves dengan phase berbeda
    let X = array![
        // Sample 1: Sine wave phase 0
        [[0.0], [0.5], [0.87], [1.0], [0.87], [0.5]],
        // Sample 2: Sine wave phase 90  
        [[1.0], [0.87], [0.5], [0.0], [-0.5], [-0.87]],
        // Sample 3: Ramp function
        [[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]],
    ].into_dyn().into_dimensionality::<ndarray::Ix3>().unwrap();

    // y: Next value prediction (shifted by 1)
    let y = array![
        // Predict next value in sequence
        [[0.5], [0.87], [1.0], [0.87], [0.5], [0.0]],   // Sample 1 shifted
        [[0.87], [0.5], [0.0], [-0.5], [-0.87], [-1.0]], // Sample 2 shifted  
        [[0.2], [0.4], [0.6], [0.8], [1.0], [1.0]],     // Sample 3 shifted
    ];

    (X, y)
}

fn main(){
    let (x,y) = create_timeseries_data();
    
    let mut model = RNN::init(x, y, 6);
    model.add_layer(1, 8, ActivationConfig::Tanh);
    model.add_layer(8, 4, ActivationConfig::ReLU);
    model.set_output_layer(1); 
    model.train(0.01, 5.0, 40, LossConfig::Huber(0.01), false, 0.0, 0, 0, "a", OptimizerConfig::SGD);
}