use std::fs::File;

use ndarray::{Array2, Array3, s};
use serde::{Deserialize, Serialize};
use bincode;
use TensorClerum::{tensor1::{PackedTensor1D, PackedTensor1DStorage}, tensor2::{PackedTensor2D, PackedTensor2DStorage}};

use super::activation::Activation;

#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub(crate) epoch: usize,
    pub(crate) loss: f32,
    pub(crate) weights: Vec<PackedTensor2DStorage>,
    pub(crate) biases: Vec<PackedTensor1DStorage>,
    pub(crate) activation_id: Vec<usize>,
    pub(crate) data_act:Vec<f32>,
    pub(crate) loss_id: usize,
    pub(crate) data_loss: f32,
    pub(crate) labels: Vec<i32>,
    pub(crate) flag: usize
}

pub fn save_checkpoint(path: &str, clr: &Checkpoint) -> std::io::Result<()> {
    let file = File::create(path)?;
    bincode::serialize_into(file, clr).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

pub fn load_checkpoint(path: &str) -> std::io::Result<Checkpoint> {
    let file = File::open(path)?;
    bincode::deserialize_from(file).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

pub fn save_checkpoint_model(
    epoch_loss: f32,
    max_core: usize,
    best_loss: &mut f32,
    epoch: usize,
    epochs: usize,
    pretrain:bool,
    count_save:&mut usize,
    w: &[&PackedTensor2D],
    b: &[&PackedTensor1D],
    act_id:&Vec<usize>,
    act_data:&Vec<f32>,
    loss_id: usize,
    data_loss:f32,
    labels: &Vec<i32>,
    path: &str,
    flag: usize
){  
    let avg_loss = epoch_loss / max_core as f32;
    let is_improved = *best_loss - avg_loss > 1e-3;
    let is_last_epoch = epoch + 1 == epochs;

    if (is_improved && !pretrain) || (is_last_epoch && !pretrain) {
        *best_loss = avg_loss;
        if *count_save >= 10 || is_last_epoch{

            let w_save = w.iter().map(|f| f.export()).collect();
            let b_save = b.iter().map(|f| f.export()).collect();

            let clr = Checkpoint {
                epoch,
                loss: *best_loss,
                weights: w_save,
                biases: b_save,
                activation_id: act_id.clone(),
                data_act: act_data.clone(),
                loss_id,
                data_loss,
                labels: labels.to_vec(),
                flag: flag
            };
        println!("Checkpoint saved at epoch {} with loss {:.6}", epoch+1, avg_loss);
        if let Err(e) = save_checkpoint(path, &clr) {
            eprintln!("Failed to save checkpoint: {:?}", e);
        }
        *count_save = 0;

        }
    }
}

pub enum Input_X {
    Array2(Array2<f32>),
    Array3(Array3<f32>),
}

fn FuncInput(config:Input_X) -> (Array2<f32>, Array3<f32>){
    match config {
        Input_X::Array2(array_base) => (array_base, Array3::zeros((0,0,0))),
        Input_X::Array3(array_base) => (Array2::zeros((0,0)), array_base),
    }
}

pub fn run_cler(checkpoint_path: &str, input: Input_X, confidence_threshold: f32) -> Option<(usize, f32)> {
    let mut clr = load_checkpoint(checkpoint_path).expect("Failed to load checkpoint");
    let (input_2D,input_3D) = FuncInput(input) ;

    if clr.flag == 1{
        let weight  = PackedTensor2D::import(clr.weights.remove(0));
        let biases  = PackedTensor1D::import(clr.biases.remove(0));
        let layer = weight.len();
        let mut model: Array2<f32> = input_2D.clone();

        for i in 0..layer {
            model = model.dot(&weight.get(i)) + &biases.get(i);
            let act = Activation::from_id(clr.activation_id[i], false, clr.data_act[i]);
            model = act.activate(model.view());
        }

        let output = model.row(0).to_owned();

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
    } else if clr.flag == 2 {
        let w_xh = PackedTensor2D::import(clr.weights.remove(0));
        let w_hh = PackedTensor2D::import(clr.weights.remove(0));
        let w_hy = PackedTensor2D::import(clr.weights.remove(0));
        let b_h  = PackedTensor1D::import(clr.biases.remove(0));
        let b_y  = PackedTensor1D::import(clr.biases.remove(0));
        let layer = w_xh.len();
        let num_layers = w_hh.len();
        let batch_size = input_3D.shape()[0];
        let seq_len = input_3D.shape()[1];

        let mut outputs = vec![Array2::zeros((batch_size,b_y.get(0).dim())); seq_len];
        let mut hidden_states = Vec::new();
        for l in 0..num_layers {
            let layer_hidden_states = vec![Array2::zeros((batch_size, w_hh.get(l).dim().0)); seq_len];
            hidden_states.push(layer_hidden_states);
        }


        for t in 0..layer {
            let mut x_t = input_3D.slice(s![.., t, ..]);

            for l in 0..num_layers {
                let h_prev = if t > 0 {&hidden_states[l][t-1]} else {&Array2::zeros(hidden_states[l][t].dim())};
                let wxh_term = &x_t.dot(&w_xh.get(l).t()); 
                let whh_term = &h_prev.dot(&w_hh.get(l).t());
                let h_t = wxh_term + whh_term + &b_h.get(l);
                let act = Activation::from_id(clr.activation_id[l], false, clr.data_act[l]);
                let h_t_activated = act.activate(h_t.view());
                hidden_states[l][t] = h_t_activated;
                x_t = hidden_states[l][t].view();

            }
            let y_t = x_t.dot(&w_hy.get(0).t()) + b_y.get(0);
            outputs[t] = y_t;
            // println!("Time {}: {:?}", t, outputs[t].shape());
        }
        let batch = outputs[0].dim().0;
        let features = outputs[0].dim().1;
        let steps = outputs.len();

        let mut flat = Vec::with_capacity(batch * steps * features);

        for t in 0..steps {
            let out_t = outputs[t].as_slice().unwrap();
            flat.extend_from_slice(out_t);
        }

        let y_pred = Array3::from_shape_vec((batch, steps, features), flat).unwrap();
        println!("y_pred {}", y_pred);
        None
    }else {
        None
    }

}