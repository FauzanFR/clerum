use TensorClerum::{tensor1::PackedTensor1D, tensor2::PackedTensor2D};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis, s};
use std::ops::AddAssign;

use crate::{activation::{Activation, ActivationConfig}, file::save_checkpoint_model, helper::{rand_arr1d, rand_arr2d}, loss::{LossConfig, LossMode}, optimizer::{OptimizerConfig, init_optimizer}};

#[derive(Debug, Clone, Copy)]
pub enum RNNTask {
    SequenceToVector,   // y_2d digunakan, y_3d kosong
    SequenceToSequence, // y_3d digunakan, y_2d kosong
}

pub enum Input_y {
    Array2(Array2<f32>),
    Array3(Array3<f32>),
}

impl From<Array2<f32>> for Input_y {
    fn from(arr: Array2<f32>) -> Self {
        Input_y::Array2(arr)
    }
}

impl From<Array3<f32>> for Input_y {
    fn from(arr: Array3<f32>) -> Self {
        Input_y::Array3(arr)
    }
}

pub struct RNN {
    X:Array3<f32>,
    y_2d: Array2<f32>,
    y_3d: Array3<f32>,
    task_type: RNNTask,
    w_xh: PackedTensor2D,  // Weight input->hidden
    w_hh: PackedTensor2D,  // Weight hidden->hidden  
    w_hy: PackedTensor2D,  // Weight hidden->output
    b_h: PackedTensor1D,   // Bias hidden
    b_y: PackedTensor1D,   // Bias output
    input_dim: Vec<usize>,
    output_dim: Vec<usize>,
    activations: Vec<usize>,
    data_act: Vec<f32>,
    labels: Vec<i32>,
    sequence_length: usize,
}

impl RNN {
    pub fn init<I>(X:Array3<f32>, y: I, sequence_length: Option<usize>) -> Self 
    where I: Into<Input_y>
    {
        let sequence_length = match sequence_length {
            Some(v) => v,
            None => X.shape()[1]
        };

        let (y_2d, y_3d, task_type) = match y.into() {
            Input_y::Array2(x) => (x, Array3::zeros((0, 0, 0)), RNNTask::SequenceToVector),
            Input_y::Array3(x) => (Array2::zeros((0, 0)), x, RNNTask::SequenceToSequence),
        };

        Self {
            X,
            y_2d,
            y_3d,
            task_type,
            w_xh: PackedTensor2D::new(),
            w_hh: PackedTensor2D::new(),
            w_hy: PackedTensor2D::new(),
            b_h: PackedTensor1D::new(),
            b_y: PackedTensor1D::new(),
            input_dim: Vec::new(),
            output_dim: Vec::new(),
            activations: Vec::new(),
            data_act: Vec::new(),
            labels: Vec::new(),
            sequence_length,
        }
    }

    pub fn add_layer(&mut self, input_dim: usize, hidden_dim: usize, activation: ActivationConfig) {
        self.w_xh.push(rand_arr2d(hidden_dim, input_dim));
        self.w_hh.push(rand_arr2d(hidden_dim, hidden_dim)); 
        self.b_h.push(rand_arr1d(hidden_dim));
        
        self.input_dim.push(input_dim);
        self.output_dim.push(hidden_dim);
        
        let (act_id, data_activation) = activation.activation_id();
        self.activations.push(act_id);
        self.data_act.push(data_activation);

        if self.input_dim.len() == 1 {
            let actual_input_dim = self.X.shape()[2];
            assert_eq!(input_dim, actual_input_dim, "First layer input_dim should be {}, but got {}", actual_input_dim, input_dim);
        }
    }

    pub fn add_labels(&mut self, labels: Vec<i32>) {
        self.labels = labels;
    }

    pub fn train(
        &mut self,
        lr: f32,
        max_norm: f32,
        epoch: usize,
        loss_mode: LossConfig,
        pretrain: bool,
        pretrain_ratio: f32,
        pretrain_epochs: usize,
        path: &str,
        optimizer: OptimizerConfig
    ){
        self.b_h.process();
        self.w_xh.process();
        self.w_hh.process();
        self.w_hy.push(self.w_hh.get(self.w_hh.len() -1).to_owned());
        self.b_y.push(self.b_h.get(self.b_h.len() -1).to_owned());
        self.w_hy.process();
        self.b_y.process();
        let loss_id = LossConfig::loss_id(loss_mode);

        if pretrain {
            println!("pretrain");
            let N: usize = (self.X.shape()[0] as f32 * pretrain_ratio).round() as usize;
            self.train_core(loss_id, pretrain_epochs, lr, max_norm, &optimizer, N, path, true);
        }

        println!("train");
        self.train_core(loss_id, epoch, lr, max_norm, &optimizer, self.X.shape()[0], path, false);
    }

    fn calculate_loss(task_type:&RNNTask, outputs: &[Array2<f32>], loss_func: &LossMode, dz_func:&LossMode, y_2d: &ArrayView2<f32>, y_3d: &ArrayView3<f32>) -> (f32, Vec<Array2<f32>>) {
        match task_type {
            RNNTask::SequenceToVector => RNN::calculate_loss_2d(outputs, y_2d, loss_func, dz_func),
            RNNTask::SequenceToSequence => RNN::calculate_loss_3d(outputs, y_3d, loss_func, dz_func),
        }
    }

    fn calculate_loss_2d(outputs: &[Array2<f32>], y_2d: &ArrayView2<f32>, loss_func: &LossMode, dz_func: &LossMode) -> (f32, Vec<Array2<f32>>) {
        let y_pred_last = outputs.last().unwrap();
        let loss = loss_func.loss(&y_2d.view(), &y_pred_last.view());
        
        let mut gradients = vec![Array2::zeros(y_pred_last.raw_dim()); outputs.len() - 1];
        let dy_last = dz_func.dz(&y_2d.view(), &y_pred_last.view(), y_2d.shape()[0] as f32);
        gradients.push(dy_last);
        
        (loss, gradients)
    }

    fn calculate_loss_3d(outputs: &[Array2<f32>], y_3d: &ArrayView3<f32>, loss_func: &LossMode, dz_func: &LossMode) -> (f32, Vec<Array2<f32>>) {
        let batch_size = outputs[0].shape()[0];
        let seq_len = outputs.len();
        let output_dim = outputs[0].shape()[1];
        
        let mut y_true_2d = Array2::zeros((batch_size * seq_len, output_dim));
        let mut y_pred_2d = Array2::zeros((batch_size * seq_len, output_dim));
        
        for t in 0..seq_len {
            for b in 0..batch_size {
                let flat_idx = t * batch_size + b;
                y_true_2d.slice_mut(s![flat_idx, ..])
                    .assign(&y_3d.slice(s![b, t, ..]));
                y_pred_2d.slice_mut(s![flat_idx, ..])
                    .assign(&outputs[t].slice(s![b, ..]));
            }
        }
        
        let loss = loss_func.loss(&y_true_2d.view(), &y_pred_2d.view());
        let dy_2d = dz_func.dz(&y_true_2d.view(), &y_pred_2d.view(), (batch_size * seq_len) as f32);
        
        let gradients: Vec<Array2<f32>> = (0..seq_len)
            .map(|t| {
                let view = dy_2d.slice(s![t*batch_size..(t+1)*batch_size, ..]);
                view.to_owned()
            })
            .collect();
        
        (loss, gradients)
    }

    fn forward (
        X:&ArrayView3<f32>,
        w_xh:&PackedTensor2D,
        w_hh:&PackedTensor2D,
        b_h:&PackedTensor1D,
        w_hy:&PackedTensor2D,
        b_y:&PackedTensor1D,
        seq_len:usize,
        num_layers:usize,
        hidden_states:&mut Vec<Vec<Array2<f32>>>,
        act_id:&Vec<usize>,
        act_data:&Vec<f32>,
        act_deriv_activate: &dyn Fn(ArrayView2<f32>, usize, f32) -> Array2<f32> ) -> Vec<Array2<f32>>{
        let mut outputs = vec![Array2::zeros((X.dim().0,b_y.get(0).dim())); seq_len];
        for t in 0..seq_len {
            let mut x_t = X.slice(s![.., t, ..]);

            for l in 0..num_layers {
                let h_prev = if t > 0 {&hidden_states[l][t-1]} else {&Array2::zeros(hidden_states[l][t].dim())};
                let wxh_term = &x_t.dot(&w_xh.get(l).t()); 
                let whh_term = &h_prev.dot(&w_hh.get(l).t());
                let h_t = wxh_term + whh_term + &b_h.get(l);
                let h_t_activated = act_deriv_activate(h_t.view(), act_id[l], act_data[l]);
                hidden_states[l][t] = h_t_activated;
                x_t = hidden_states[l][t].view();

            }
            let y_t = x_t.dot(&w_hy.get(0).t()) + b_y.get(0);
            outputs[t] = y_t;
            // println!("Time {}: {:?}", t, outputs[t].shape());
        }
        outputs
    }

    fn backward (
        X:&ArrayView3<f32>,
        hidden_states: &Vec<Vec<Array2<f32>>>,
        dW_hy: &mut PackedTensor2D,
        dZ: &[Array2<f32>],
        db_y: &mut PackedTensor1D,
        w_hy: &PackedTensor2D,
        dh_next: &mut Vec<Array2<f32>>,
        dW_xh: &mut PackedTensor2D,
        dW_hh: &mut PackedTensor2D,
        db_h: &mut PackedTensor1D,
        w_xh: &PackedTensor2D,
        w_hh: &PackedTensor2D,
        seq_len: usize,
        num_layers: usize,
        act_id: &Vec<usize>,
        act_data: &Vec<f32>,
        act_deriv_deriv: &dyn Fn(ArrayView2<f32>, usize, f32) -> Array2<f32>
    ){  
        dW_xh.fill_all(0.0);
        dW_hh.fill_all(0.0);
        db_h.fill_all(0.0);
        dW_hy.fill_all(0.0);
        db_y.fill_all(0.0);
        for dh in &mut *dh_next {dh.fill(0.0)};

        for t in (0..seq_len).rev() {
            let h_last = &hidden_states[num_layers-1][t];
            let temp = &h_last.t().dot(&dZ[t]);
            dW_hy.get_mut(0).add_assign(&temp.t());
            db_y.get_mut(0).add_assign(&dZ[t].sum_axis(Axis(0)));
            let mut dh = dZ[t].dot(&w_hy.get(0));

            for l in (0..num_layers).rev() {
                if t < seq_len - 1 {
                    dh = &dh + &dh_next[l];
                }
                let h_current = &hidden_states[l][t];
                let dz = &dh * act_deriv_deriv(h_current.view(), act_id[l], act_data[l]);

                let input_for_layer = if l == 0 {
                    X.slice(s![.., t, ..])
                } else {
                    hidden_states[l-1][t].view()
                };

                let grad_xh = dz.t().dot(&input_for_layer);
                dW_xh.get_mut(l).add_assign(&grad_xh);

                if t > 0 {
                    let h_prev = &hidden_states[l][t-1];
                    dW_hh.get_mut(l).add_assign(&h_prev.t().dot(&dz));
                }

                db_h.get_mut(l).add_assign(&dz.sum_axis(Axis(0)));
                
                if l > 0 {
                    dh = dz.dot(&w_xh.get(l));
                }
                if t > 0 {
                    dh_next[l] = dz.dot(&w_hh.get(l));
                }

            }
        }
    }

    fn clip_gradients_by_norm(
        dW_xh: &mut PackedTensor2D,
        dW_hh: &mut PackedTensor2D,
        dW_hy: &mut PackedTensor2D, 
        db_h: &mut PackedTensor1D,
        db_y: &mut PackedTensor1D,
        max_norm: f32,
    ) -> f32 {
        let mut total_norm = 0.0;
        
        for l in 0..dW_xh.len() {
            total_norm += dW_xh.get(l).mapv(|x| x.powi(2)).sum();
            total_norm += dW_hh.get(l).mapv(|x| x.powi(2)).sum();
            total_norm += db_h.get(l).mapv(|x| x.powi(2)).sum();
        }
        
        total_norm += dW_hy.get(0).mapv(|x| x.powi(2)).sum();
        total_norm += db_y.get(0).mapv(|x| x.powi(2)).sum();
        
        let total_norm = total_norm.sqrt();
        
        if total_norm > max_norm {
            let scale_factor = max_norm / total_norm;
            
            for l in 0..dW_xh.len() {
                dW_xh.get_mut(l).mapv_inplace(|x| x * scale_factor);
                dW_hh.get_mut(l).mapv_inplace(|x| x * scale_factor);
                db_h.get_mut(l).mapv_inplace(|x| x * scale_factor);
            }
            
            dW_hy.get_mut(0).mapv_inplace(|x| x * scale_factor);
            db_y.get_mut(0).mapv_inplace(|x| x * scale_factor);
            
            println!("Gradients clipped: norm {:.4} -> {:.4}", total_norm, max_norm);
        }
        
        total_norm
    }

    fn train_core(
        &mut self,
        loss_id: (usize, f32),
        epochs: usize,
        lr: f32,
        max_norm: f32,
        optimizer: &OptimizerConfig,
        N: usize,
        path: &str,
        pretrain: bool

    ){
        let (loss_id, data_loss) = loss_id;
        let act_id:&Vec<usize> = &self.activations;
        let act_data:&Vec<f32> = &self.data_act;

        let mut best_loss:f32 = 9999.9;
        let mut count_save:usize = 10;

        let X = &self.X.slice(s![0..N, .., ..]);
        let y_3d = if self.y_3d.shape()[0] > 0{ &self.y_3d.slice(s![0..N, .., ..])} else {&self.y_3d.view()};
        let y_2d = if self.y_2d.shape()[0] > 0{ &self.y_2d.slice(s![0..N, ..])} else {&self.y_2d.view()};


        let act_deriv_deriv = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, true, data).deriv(z)
        };

        let act_deriv_activate = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, false, data).activate(z)
        };

        let loss_func = LossMode::from_id(loss_id, false, data_loss);
        let dz_func = LossMode::from_id(loss_id, true, data_loss);
        
        let batch_size = X.shape()[0];
        let seq_len = self.sequence_length;
        let num_layers = self.w_hh.len();
        let mut all_hidden_states = Vec::new();
        for l in 0..num_layers {
            let layer_hidden_states = vec![Array2::zeros((batch_size, self.output_dim[l])); seq_len];
            all_hidden_states.push(layer_hidden_states);
        }
        
        let mut dW_xh = self.w_xh.copy_and_fill(0.0);
        let mut dW_hh = self.w_hh.copy_and_fill(0.0);
        let mut dW_hy = self.w_hy.copy_and_fill(0.0);
        let mut db_h = self.b_h.copy_and_fill(0.0);
        let mut db_y = self.b_y.copy_and_fill(0.0);
        let mut dh_next: Vec<Array2<f32>> = vec![Array2::zeros((batch_size, self.output_dim[0])); num_layers];
        let mut optimizer = init_optimizer(&vec![&self.w_xh, &self.w_hh, &self.w_hy],&vec![&self.b_h, &self.b_y],optimizer); 

        for epoch in 0..epochs{
            let outputs = RNN::forward(X, &self.w_xh, &self.w_hh, &self.b_h, &self.w_hy, &self.b_y, seq_len, num_layers, &mut all_hidden_states, act_id, act_data, &act_deriv_activate);
            let (loss, dZ) = RNN::calculate_loss(&self.task_type, &outputs, &loss_func, &dz_func, y_2d, y_3d);
            RNN::backward(X, &all_hidden_states, &mut dW_hy, &dZ, &mut db_y, &self.w_hy, &mut dh_next, &mut dW_xh, &mut dW_hh, &mut db_h, &self.w_xh, &self.w_hh, seq_len, num_layers, act_id, act_data, &act_deriv_deriv);
            RNN::clip_gradients_by_norm(&mut dW_xh, &mut dW_hh, &mut dW_hy, &mut db_h, &mut db_y, max_norm);
            optimizer.run(&mut vec![&mut self.w_xh, &mut self.w_hh, &mut self.w_hy],&mut vec![&mut self.b_h, &mut self.b_y], &vec![&dW_xh, &dW_hh, &dW_hy],&vec![&db_h, &db_y], lr);


            println!("{}", loss);
            save_checkpoint_model(loss, 1, &mut best_loss, epoch, epochs, pretrain, &mut count_save, &vec![&self.w_xh, &self.w_hh, &self.w_hy], &vec![&self.b_h, &self.b_y], act_id, act_data, loss_id, data_loss, &self.labels, path, 2);
        }
    }
}