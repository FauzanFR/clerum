use std::fs::File;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use bincode;

use super::activation::Activation;

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub(crate) epoch: usize,
    pub(crate) loss: f32,
    pub(crate) weights: Vec<Array2<f32>>,
    pub(crate) biases: Vec<Array1<f32>>,
    pub(crate) activation_id: Vec<usize>,
    pub(crate) data_act:Vec<f32>,
    pub(crate) loss_id: usize,
    pub(crate) data_loss: f32,
    pub(crate) labels: Vec<i32>
}

pub fn save_checkpoint(path: &str, clr: &Checkpoint) -> std::io::Result<()> {
    let file = File::create(path)?;
    bincode::serialize_into(file, clr).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

pub fn load_checkpoint(path: &str) -> std::io::Result<Checkpoint> {
    let file = File::open(path)?;
    bincode::deserialize_from(file).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}


pub fn run_cler(checkpoint_path: &str, input: Array2<f32>, confidence_threshold: f32) -> Option<(usize, f32)> {
    let clr = load_checkpoint(checkpoint_path).expect("Failed to load checkpoint");
    let mut X = input;
    let layer = clr.weights.len();

    for i in 0..layer {
        X = X.dot(&clr.weights[i]) + &clr.biases[i];
        let act = Activation::from_id(clr.activation_id[i], false, clr.data_act[i]);
        X = act.activate(X.view());
    }

    let output = X.row(0).to_owned();

    let maybe_best = output.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((idx, &score)) = maybe_best {
        let last_act: Activation = Activation::from_id(*clr.activation_id.last().unwrap(), false, *clr.data_act.last().unwrap());
        let is_probabilistic = matches!(last_act, Activation::Softmax);

        if is_probabilistic {
            if clr.labels.len() == 0 {
                println!("Predicted index: {} | Confidence: {:.2}%", idx, score * 100.0);

            } else{
                let label = clr.labels[idx];
                println!("Predicted index: {} | Confidence: {:.2}% | answer {}", idx, score * 100.0, label);
            }

            if score >= confidence_threshold {
                Some((idx, score))
            } else {
                None
            }
        } else {
            println!("Predicted index: {} | Output score: {:.4}", idx, score);
            Some((idx, score))
        }
    } else {
        None
    }
}