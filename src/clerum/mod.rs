#![allow(non_snake_case, non_camel_case_types)]


pub mod activation;
pub mod file;
pub mod helper;
pub mod loss;
pub mod optimizer;

use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayViewMut2, Axis, Ix2};
use std::ops::AddAssign;
use rayon::prelude::*;
use TensorClerum::{tensor1::PackedTensor1D, tensor2::PackedTensor2D};

use crate::{clerum::{activation::{Activation, ActivationConfig}, file::{save_checkpoint, Checkpoint}, helper::{clr_format, rand_arr2d, split_range}, loss::{LossConfig, LossMode}, optimizer::{Adam, Momentum, Optimizer, OptimizerConfig, RMSprop}}, file::load_checkpoint, helper::rand_arr1d};
pub struct FNN{
    X:Array2<f32>,
    y:Array2<f32>,
    w: PackedTensor2D,
    b: PackedTensor1D,
    input_dim: Vec<usize>,
    output_dim: Vec<usize>,
    activations: Vec<usize>,
    data_act: Vec<f32>,
    labels: Vec<i32>,
}

impl FNN {
    pub fn init(X:Array2<f32>, y:Array2<f32>) -> Self {
        Self{
            X: X, y: y, w:PackedTensor2D::new(), b:PackedTensor1D::new(),
            input_dim:Vec::new(), output_dim:Vec::new(),
            activations:Vec::new(), data_act:Vec::new(),
            labels: Vec::new()
        }
    }

    pub fn from_checkpoint(&mut self, path:&str) {
        let checkpoint = load_checkpoint(path).expect("Failed to load checkpoint");

        self.w = PackedTensor2D::import(checkpoint.weights);
        self.b = PackedTensor1D::import(checkpoint.biases);
        self.activations = checkpoint.activation_id;
        self.data_act = checkpoint.data_act;
        self.labels = checkpoint.labels;

        for i in 0..self.w.len(){
            let (x, y) = self.w.dim(i);
            self.input_dim.push(x);
            self.output_dim.push(y);
        }
    }

    pub fn add_layer(&mut self, input_dim:usize, output_dim:usize, activation:ActivationConfig) {
        self.w.push(rand_arr2d( output_dim, input_dim));
        self.b.push(rand_arr1d(output_dim));
        self.input_dim.push(input_dim);
        self.output_dim.push(output_dim);
        let (act_id, data_activation) = ActivationConfig::activation_id(activation);
        self.activations.push(act_id);
        self.data_act.push(data_activation);
    }

    pub fn add_labels(&mut self, labels: Vec<i32>) {
        self.labels = labels;
    }

    fn build_forwrd_buffer(&self, N:usize, max_core:usize) -> Vec<Vec<Array2<f32>>> {
        let mut buffers = Vec::with_capacity(1+self.output_dim.len());
        buffers.push(Array2::zeros((N, self.input_dim[0])));

        for &out_dim in &self.output_dim{
            buffers.push(Array2::zeros((N, out_dim)));
        }
        (0..max_core).map(|_| buffers.iter().map(|arr| arr.clone()).collect()).collect()
    }

    fn build_z_cache(&self, N:usize, max_core:usize) -> Vec<Vec<Array2<f32>>>{
        let mut buffers:Vec<Array2<f32>> = Vec::new();
        for &out_dim in &self.output_dim{
            buffers.push(Array2::zeros((N, out_dim)));
        }
        (0..max_core).map(|_| buffers.iter().map(|arr| arr.clone()).collect()).collect()
    }

    pub fn train(
        &mut self,
        lr: f32,
        epoch: usize,
        loss_mode: LossConfig,
        pretrain: bool,
        pretrain_ratio: f32,
        pretrain_epochs: usize,
        parallel_threads: usize,
        path: &str,
        optimizer: OptimizerConfig
    ) {
        self.b.process();
        self.w.process();
        let max_core = if parallel_threads>0 && parallel_threads <= rayon::current_num_threads() {parallel_threads} else {rayon::current_num_threads()};
        let loss_id = LossConfig::loss_id(loss_mode);
 
        if pretrain {
            println!("pretrain");
            let N: usize = (self.X.nrows() as f32 * pretrain_ratio).round() as usize;

            let layer = self.w.len();
            let output_dim = *self.output_dim.last().unwrap();

            let a = self.build_forwrd_buffer(N, max_core);
            let z = self.build_z_cache(N, max_core);
            let y_pred: Vec<Array2<f32>> = (0..max_core).map(|_| Array2::<f32>::zeros((N, output_dim))).collect();
            let dW_buffer = self.w.copy_and_fill(0.0);
            let db_buffer = self.b.copy_and_fill(0.0);

            self.train_core(
                layer, pretrain_epochs,
                dW_buffer, db_buffer, lr,
                z, a, y_pred, true,
                N,1, &clr_format(path),&optimizer,loss_id
            );
        }

        println!("train");
        let layer = self.w.len();

        let total = self.X.nrows();
        let output_dim = *self.output_dim.last().unwrap();
        let max_batch = (total + max_core - 1) / max_core;

        let a = self.build_forwrd_buffer(max_batch, max_core);
        let z = self.build_z_cache(max_batch, max_core);
        let y_pred:Vec<Array2<f32>> = (0..max_core).map(|_| Array2::<f32>::zeros((max_batch, output_dim))).collect();
        let dW_buffer = self.w.copy_and_fill(0.0);
        let db_buffer = self.b.copy_and_fill(0.0);


        self.train_core(
            layer, epoch,
            dW_buffer, db_buffer, lr,
            z, a, y_pred, false, 
            0,max_core, &clr_format(path),&optimizer,loss_id
        );
    }

    fn init_optimizer(w: &PackedTensor2D, b: &PackedTensor1D, config: &OptimizerConfig) -> Optimizer {
        match config {
            OptimizerConfig::SGD => Optimizer::SGD,
            OptimizerConfig::Momentum(gamma) => Optimizer::Momentum(Momentum::init(w,b,*gamma)),
            OptimizerConfig::RMSprop(gamma) => Optimizer::RMSprop(RMSprop::init(w,b, *gamma)),
            OptimizerConfig::Adam(b1, b2) => Optimizer::Adam(Adam::init(w,b,*b1, *b2,0))
        }.init(w, b)
    }
    
    fn forward_pass(
        layer: usize,
        w: &PackedTensor2D,
        b: &PackedTensor1D,
        a_batch: &mut Vec<ArrayViewMut2<f32>>,
        z_batch: &mut Vec<ArrayViewMut2<f32>>,
        y_pred: &mut ArrayViewMut2<f32>,
        act_id:&Vec<usize>,
        act_data:&Vec<f32>,
        act_deriv_activate: &dyn Fn(ArrayView2<f32>, usize, f32) -> Array2<f32>,
    ) {
        for l in 0..layer {
            z_batch[l].assign(&a_batch[l].dot(&w.get(l)));
            z_batch[l] += &b.get(l);
            if l < layer - 1 {
                a_batch[l + 1].assign(&act_deriv_activate(z_batch[l].view(), act_id[l], act_data[l]));
            } else {
                y_pred.assign(&z_batch[l]);
            }
        }
    }

    fn backward_pass(
        layer: usize,
        w: &PackedTensor2D,
        a_batch: &mut Vec<ArrayViewMut2<f32>>,
        z_batch: &mut Vec<ArrayViewMut2<f32>>,
        local_dW: &mut Vec<Array2<f32>>,
        local_db: &mut Vec<Array1<f32>>,
        act_id:&Vec<usize>,
        act_data:&Vec<f32>,
        dZ: &mut Array2<f32>,
        act_deriv_deriv: &dyn Fn(ArrayView2<f32>, usize, f32) -> Array2<f32>,
    ){
        for l in (0..layer).rev() {
            local_dW[l] = a_batch[l].t().dot(dZ);
            local_db[l] = dZ.sum_axis(Axis(0));

            if l == 0 { break }

            let dA = dZ.dot(&w.get(l).t());
            let grad = act_deriv_deriv(z_batch[l - 1].view(), act_id[l - 1], act_data[l-1]);
            *dZ = &dA * &grad;
        }
    }

    fn maybe_save_checkpoint(
        epoch_loss: f32,
        max_core: usize,
        best_loss: &mut f32,
        epoch: usize,
        epochs: usize,
        pretrain:bool,
        count_save:&mut usize,
        w: &PackedTensor2D,
        b: &PackedTensor1D,
        act_id:&Vec<usize>,
        act_data:&Vec<f32>,
        loss_id: usize,
        data_loss:f32,
        labels: &Vec<i32>,
        path: &str
    ){  
        let avg_loss = epoch_loss / max_core as f32;
        let is_improved = *best_loss - avg_loss > 1e-3;
        let is_last_epoch = epoch + 1 == epochs;

        if (is_improved && !pretrain) || (is_last_epoch && !pretrain) {
            *best_loss = avg_loss;
            if *count_save >= 10 || is_last_epoch{
                let clr = Checkpoint {
                    epoch,
                    loss: *best_loss,
                    weights: w.export(),
                    biases: b.export(),
                    activation_id: act_id.clone(),
                    data_act: act_data.clone(),
                    loss_id,
                    data_loss,
                    labels: labels.to_vec(),
                };
            println!("Checkpoint saved at epoch {} with loss {:.6}", epoch+1, avg_loss);
            if let Err(e) = save_checkpoint(path, &clr) {
                eprintln!("Failed to save checkpoint: {:?}", e);
            }
            *count_save = 0;

            }
        }
    }

    fn train_core(
        &mut self,
        layer:usize,
        epochs:usize,
        mut dW_buffer:PackedTensor2D,
        mut db_buffer:PackedTensor1D,
        lr:f32,
        mut z_cache_all_batch:Vec<Vec<Array2<f32>>>,
        mut a_cache_all_batch:Vec<Vec<Array2<f32>>>,
        mut y_pred_all_batch: Vec<Array2<f32>>,
        pretrain: bool,
        N:usize,
        max_core:usize,
        path: &str,
        optimizer: &OptimizerConfig,
        loss_id: (usize, f32)
    )
    {
        let w= &mut self.w;
        let b= &mut self.b;
        
        let (loss_id, data_loss) = loss_id;
        let act_id:&Vec<usize> = &self.activations;
        let act_data:&Vec<f32> = &self.data_act;

        let mut best_loss:f32 = 9999.9;
        let mut count_save:usize = 10;

        let (y_all_batch, X_all_batch) = if pretrain {
            (&self.y.slice(s![0..N, ..]), &self.X.slice(s![0..N, ..]))
        } else {
            (&self.y.view(), &self.X.view())
        };

        let act_deriv_deriv = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, true, data).deriv(z)
        };
        let act_deriv_activate = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, false, data).activate(z)
        };

        let loss_func = LossMode::from_id(loss_id, false, data_loss);
        let dz_func = LossMode::from_id(loss_id, true, data_loss);
        let total = self.X.nrows();

        let mut optimizer = FNN::init_optimizer(&w,&b,optimizer); 

        for epoch in 0..epochs{
            let chunks = split_range(total, max_core);
            let mut epoch_loss = 0.0;

            let results:Vec<_> = chunks.into_par_iter()
            .zip(a_cache_all_batch.par_iter_mut())
            .zip(z_cache_all_batch.par_iter_mut())
            .zip(y_pred_all_batch.par_iter_mut())
            .map(|(((chunk, a_cache), z_cache), y_pred)| {
                let (start, end) = chunk;
                let end1 = if pretrain { N } else { end };
                let end2 = if pretrain { N } else { end - start };
                let n =(end - start) as f32;

                let mut a_batch = a_cache.iter_mut().map(|a| a.slice_mut(s![0..end2, ..])).collect::<Vec<_>>();
                let mut z_batch = z_cache.iter_mut().map(|a| a.slice_mut(s![0..end2, ..])).collect::<Vec<_>>();
                let mut y_pred = y_pred.slice_mut(s![0..end2, ..]);

                let X_batch = X_all_batch.slice(s![start..end1, ..]);
                let y_batch = y_all_batch.slice(s![start..end1, ..]);

                a_batch[0].assign(&X_batch);
                
                FNN::forward_pass(layer, w, b, &mut a_batch, &mut z_batch, &mut y_pred, act_id, act_data, &act_deriv_activate);

                let loss = loss_func.loss(&y_batch, &y_pred.view());
                let mut dZ = dz_func.dz(&y_batch, &y_pred.view(), n);

                let mut local_dW = vec![Array2::zeros(w.dim(0)); layer];
                let mut local_db = vec![Array1::zeros(b.dim(0)); layer];

                FNN::backward_pass(layer, w, &mut a_batch, &mut z_batch, &mut local_dW, &mut local_db, act_id, act_data, &mut dZ, &act_deriv_deriv);

                (loss, local_dW, local_db)
                
            }).collect();
            
            dW_buffer.fill_all(0.0);
            db_buffer.fill_all(0.0);

            for (loss, local_dW, local_db) in results{
                epoch_loss += loss;
                for i in 0..layer {
                    dW_buffer.get_mut(i).add_assign(&local_dW[i]);
                    db_buffer.get_mut(i).add_assign(&local_db[i]);
                }
            }

            optimizer.run(w, b, &dW_buffer, &db_buffer, lr);

            FNN::maybe_save_checkpoint(epoch_loss, max_core, &mut best_loss, epoch, epochs, pretrain, &mut count_save, w, b, act_id, act_data, loss_id, data_loss, &self.labels, path);
            count_save +=1;

            let grad_norm = dW_buffer.iter().map(|dw| dw.mapv(|x| x.powi(2)).sum()).sum::<f32>().sqrt();
            println!(
                "Epoch {} | Loss: {:.6} | LR: {:.4e} | Grad: {:.4e}",
                epoch + 1,
                epoch_loss / max_core as f32,
                lr,
                grad_norm
            );
        }
    }
    
}

#[derive(Debug, Clone, Copy)]
pub enum RNNTask {
    SequenceToVector,   // y_2d digunakan, y_3d kosong
    SequenceToSequence, // y_3d digunakan, y_2d kosong
}

enum Input_y {
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
    hidden_dim: Vec<usize>,
    output_dim: usize,
    activations: Vec<usize>,
    data_act: Vec<f32>,
    labels: Vec<i32>,
    sequence_length: usize,
}



impl RNN {
    pub fn init<I>(X:Array3<f32>, y: I, sequence_length: usize) -> Self 
    where I: Into<Input_y>
    {

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
            hidden_dim: Vec::new(),
            output_dim: 0,
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
        self.hidden_dim.push(hidden_dim);
        
        let (act_id, data_activation) = activation.activation_id();
        self.activations.push(act_id);
        self.data_act.push(data_activation);

        if self.input_dim.len() == 1 {
            let actual_input_dim = self.X.shape()[2];
            assert_eq!(input_dim, actual_input_dim, "First layer input_dim should be {}, but got {}", actual_input_dim, input_dim);
        }
    }

    pub fn set_output_layer(&mut self, output_dim: usize) {
        let last_hidden_dim = *self.hidden_dim.last().expect("Must add at least one hidden layer first!");
        self.w_hy.push(rand_arr2d(output_dim, last_hidden_dim));
        self.b_y.push(rand_arr1d(output_dim));
        self.output_dim = output_dim;
    }

    pub fn add_labels(&mut self, labels: Vec<i32>) {
        self.labels = labels;
    }

    fn build_forward_buffer(&self, batch_size: usize, max_core: usize) -> (Vec<Vec<Array2<f32>>>, Vec<Vec<Array2<f32>>>) {
        // Buffer untuk hidden states (per time step)
        let mut hidden_buffers = Vec::with_capacity(max_core);
        let mut output_buffers = Vec::with_capacity(max_core);

        for _ in 0..max_core {
            let mut core_hidden = Vec::with_capacity(self.sequence_length + 1);
            let mut core_output = Vec::with_capacity(self.sequence_length);
            
            // Inisialisasi hidden state awal (h0)
            for layer in 0..self.hidden_dim.len() {
                core_hidden.push(Array2::zeros((batch_size, self.hidden_dim[layer])));
            }
            
            // Buffer output untuk setiap time step
            for _ in 0..self.sequence_length {
                core_output.push(Array2::zeros((batch_size, self.output_dim)));
            }
            
            hidden_buffers.push(core_hidden);
            output_buffers.push(core_output);
        }

        (hidden_buffers, output_buffers)
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
        parallel_threads: usize,
        path: &str,
        optimizer: OptimizerConfig
    ){
        self.b_h.process();
        self.w_xh.process();
        self.w_hh.process();
        self.w_hy.process();
        self.b_y.process();

        let loss_id = LossConfig::loss_id(loss_mode);

        let max_core = if parallel_threads > 0 && parallel_threads <= rayon::current_num_threads() {
            parallel_threads
        } else {
            rayon::current_num_threads()
        };
    self.train_core(loss_id, epoch, lr, max_norm);
    }

    fn calculate_loss(task_type:&RNNTask, outputs: &[Array2<f32>], loss_func: &LossMode, dz_func:&LossMode, y_2d: &Array2<f32>, y_3d: &Array3<f32>) -> (f32, Vec<Array2<f32>>) {
        match task_type {
            RNNTask::SequenceToVector => RNN::calculate_loss_2d(outputs, y_2d, loss_func, dz_func),
            RNNTask::SequenceToSequence => RNN::calculate_loss_3d(outputs, y_3d, loss_func, dz_func),
        }
    }

    fn calculate_loss_2d(outputs: &[Array2<f32>], y_2d: &Array2<f32>, loss_func: &LossMode, dz_func: &LossMode) -> (f32, Vec<Array2<f32>>) {
        let y_pred_last = outputs.last().unwrap();
        let loss = loss_func.loss(&y_2d.view(), &y_pred_last.view());
        
        let mut gradients = vec![Array2::zeros(y_pred_last.raw_dim()); outputs.len() - 1];
        let dy_last = dz_func.dz(&y_2d.view(), &y_pred_last.view(), y_2d.shape()[0] as f32);
        gradients.push(dy_last);
        
        (loss, gradients)
    }

    fn calculate_loss_3d(outputs: &[Array2<f32>], y_3d: &Array3<f32>, loss_func: &LossMode, dz_func: &LossMode) -> (f32, Vec<Array2<f32>>) {
        // Reshape logic yang tadi kita bahas
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
        
        let mut gradients = Vec::new();
        for t in 0..seq_len {
            let mut dy_t = Array2::zeros((batch_size, output_dim));
            for b in 0..batch_size {
                let flat_idx = t * batch_size + b;
                dy_t.slice_mut(s![b, ..])
                    .assign(&dy_2d.slice(s![flat_idx, ..]));
            }
            gradients.push(dy_t);
        }
        
        (loss, gradients)
    }

    fn forward (
        X:&Array3<f32>,
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
        let mut outputs = Vec::new();
        for t in 0..seq_len {
            let mut x_t = X.slice(s![.., t, ..]);

            for l in 0..num_layers {
                let h_prev = if t > 0 {&hidden_states[l][t-1]} else {&Array2::zeros(hidden_states[l][t].dim())};
                let wxh_term = &x_t.dot(&w_xh.get(l));
                let whh_term = &h_prev.dot(&w_hh.get(l).t());
                let h_t = wxh_term + whh_term + &b_h.get(l);
                let h_t_activated = act_deriv_activate(h_t.view(), act_id[l], act_data[l]);
                hidden_states[l][t] = h_t_activated;
                x_t = hidden_states[l][t].view();

            }
            let y_t = x_t.dot(&w_hy.get(0).t()) + b_y.get(0);
            outputs.push(y_t);
            // println!("Time {}: {:?}", t, outputs[t].shape());
        }
        outputs
    }

    fn backward (
        X:&Array3<f32>,
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
                dW_xh.get_mut(l).add_assign(&grad_xh.t());

                if t > 0 {
                    let h_prev = &hidden_states[l][t-1];
                    let sum = &dW_hh.get(l) + &h_prev.t().dot(&dz);
                    dW_hh.get_mut(l).assign(&sum);
                }

                let sum = &db_h.get(l) + &dz.sum_axis(Axis(0));
                db_h.get_mut(l).assign(&sum);
                
                if l > 0 {
                    dh = dz.dot(&w_xh.get(l).t());
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
            
            // println!("Gradients clipped: norm {:.4} -> {:.4}", total_norm, max_norm);
        }
        
        total_norm
    }

    fn sgd_update(
        w_xh: &mut PackedTensor2D,
        w_hh: &mut PackedTensor2D,
        b_h: &mut PackedTensor1D,
        w_hy: &mut PackedTensor2D,
        b_y: &mut PackedTensor1D,
        dW_xh: &PackedTensor2D,
        dW_hh: &PackedTensor2D, 
        dW_hy: &PackedTensor2D,
        db_h: &PackedTensor1D,
        db_y: &PackedTensor1D,
        lr: f32,
    ) {
        for l in 0..w_xh.len() {
            w_xh.get_mut(l).scaled_add(-lr, &dW_xh.get(l));
            w_hh.get_mut(l).scaled_add(-lr, &dW_hh.get(l)); 
            b_h.get_mut(l).scaled_add(-lr, &db_h.get(l));
        }
        
        w_hy.get_mut(0).scaled_add(-lr, &dW_hy.get(0));
        b_y.get_mut(0).scaled_add(-lr, &db_y.get(00));
        
        // println!("SGD update applied, lr: {:.6}", lr);
    }

    fn train_core(
        &mut self,
        loss_id: (usize, f32),
        epoch: usize,
        lr: f32,
        max_norm: f32
    ){
        let (loss_id, data_loss) = loss_id;
        let act_id:&Vec<usize> = &self.activations;
        let act_data:&Vec<f32> = &self.data_act;
        
        let act_deriv_deriv = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, true, data).deriv(z)
        };

        let act_deriv_activate = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, false, data).activate(z)
        };

        let loss_func = LossMode::from_id(loss_id, false, data_loss);
        let dz_func = LossMode::from_id(loss_id, true, data_loss);
        
        let batch_size = self.X.shape()[0];
        let seq_len = self.X.shape()[1];
        let num_layers = self.w_hh.len();
        let mut all_hidden_states = Vec::new();
        for l in 0..num_layers {
            let layer_hidden_states = vec![Array2::zeros((batch_size, self.hidden_dim[l])); seq_len];
            all_hidden_states.push(layer_hidden_states);
        }
        
        let mut dW_xh = self.w_xh.copy_and_fill(0.0);
        let mut dW_hh = self.w_hh.copy_and_fill(0.0);
        let mut dW_hy = self.w_hy.copy_and_fill(0.0);
        let mut db_h = self.b_h.copy_and_fill(0.0);
        let mut db_y = self.b_y.copy_and_fill(0.0);
        let mut dh_next: Vec<Array2<f32>> = vec![Array2::zeros((batch_size, self.hidden_dim[0])); num_layers];

        for _ in 0..epoch{
            let outputs = RNN::forward(&self.X, &self.w_xh, &self.w_hh, &self.b_h, &self.w_hy, &self.b_y, seq_len, num_layers, &mut all_hidden_states, act_id, act_data, &act_deriv_activate);
            let (loss, dZ) = RNN::calculate_loss(&self.task_type, &outputs, &loss_func, &dz_func, &self.y_2d, &self.y_3d);
            RNN::backward(&self.X, &all_hidden_states, &mut dW_hy, &dZ, &mut db_y, &self.w_hy, &mut dh_next, &mut dW_xh, &mut dW_hh, &mut db_h, &self.w_xh, &self.w_hh, seq_len, num_layers, act_id, act_data, &act_deriv_deriv);
            RNN::clip_gradients_by_norm(&mut dW_xh, &mut dW_hh, &mut dW_hy, &mut db_h, &mut db_y, max_norm);
            RNN::sgd_update(&mut self.w_xh, &mut self.w_hh, &mut self.b_h, &mut self.w_hy, &mut self.b_y, &dW_xh, &dW_hh, &dW_hy, &db_h, &db_y, lr);
            println!("{}", loss)
        }

        
    
    }

}