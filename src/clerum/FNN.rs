use TensorClerum::{tensor1::PackedTensor1D, tensor2::PackedTensor2D};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, s};

use crate::{activation::{Activation, ActivationConfig}, file::{load_checkpoint, save_checkpoint_model}, helper::{clr_format, rand_arr1d, rand_arr2d, split_range}, loss::{LossConfig, LossMode}, optimizer::{OptimizerConfig, init_optimizer}};
use rayon::prelude::*;
use std::ops::AddAssign;

pub enum PretrainConfig {
    Disabled,
    Partial{
        data_ratio: f32,
        epochs: usize,      
    }
}

fn FuncPretrainConfig(config : PretrainConfig) -> (f32, usize){
    match config {
        PretrainConfig::Disabled => (0.0, 0),
        PretrainConfig::Partial { data_ratio, epochs } => (data_ratio, epochs),
    }
}

pub enum BatchConfig {
    Sequential,
    Parallel {
        parallel_threads: usize,
        batch_size: usize,
    }
}

fn FuncBatchConfig(config : BatchConfig) -> (usize, usize){
    match config {
        BatchConfig::Sequential => (1, 0),
        BatchConfig::Parallel { parallel_threads, batch_size } => (parallel_threads, batch_size),
    }
}

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
        let mut checkpoint = load_checkpoint(path).expect("Failed to load checkpoint");
        if checkpoint.flag == 1{
            self.w = PackedTensor2D::import(checkpoint.weights.remove(0));
            self.b = PackedTensor1D::import(checkpoint.biases.remove(0));
            self.activations = checkpoint.activation_id;
            self.data_act = checkpoint.data_act;
            self.labels = checkpoint.labels;

            for i in 0..self.w.len(){
                let (x, y) = self.w.dim(i);
                self.input_dim.push(x);
                self.output_dim.push(y);
            }
        }else {
            panic!("fatal error: wrong model! code: {}", checkpoint.flag)
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

    fn build_forward_buffer(&self, N:usize, max_core:usize) -> Vec<Vec<Array2<f32>>> {
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
        PretrainConfig: PretrainConfig,
        BatchConfig: BatchConfig,
        path: &str,
        optimizer: OptimizerConfig
    ) {
        self.b.process();
        self.w.process();
        
        let(mut data_ratio, pretrain_epochs) = FuncPretrainConfig(PretrainConfig);
        let(parallel_threads,mut batch_size) = FuncBatchConfig(BatchConfig);

        batch_size = if batch_size > 0 && !parallel_threads*batch_size>self.X.nrows() {batch_size} else {self.X.nrows()/parallel_threads as usize};
        let max_core = if parallel_threads>0 && parallel_threads <= rayon::current_num_threads() {parallel_threads} else {rayon::current_num_threads()};
        let loss_id = LossConfig::loss_id(loss_mode);
 
        if data_ratio > 0.0 && pretrain_epochs > 0 {
            println!("pretrain");

            data_ratio = data_ratio.min(1.0);
            let N: usize = (self.X.nrows() as f32 * data_ratio).round() as usize;

            let layer = self.w.len();
            let output_dim = *self.output_dim.last().unwrap();

            let a = self.build_forward_buffer(N, max_core);
            let z = self.build_z_cache(N, max_core);
            let y_pred: Vec<Array2<f32>> = (0..max_core).map(|_| Array2::<f32>::zeros((N, output_dim))).collect();
            let dW_buffer = self.w.copy_and_fill(0.0);
            let db_buffer = self.b.copy_and_fill(0.0);

            self.train_core(
                layer, pretrain_epochs,
                dW_buffer, db_buffer, lr,
                z, a, y_pred, true,
                N,1, &clr_format(path),&optimizer,loss_id, N
            );
        }

        println!("train");
        let layer = self.w.len();

        let total = self.X.nrows();
        let output_dim = *self.output_dim.last().unwrap();
        let max_batch = (total + max_core - 1) / max_core;

        let a = self.build_forward_buffer(max_batch, max_core);
        let z = self.build_z_cache(max_batch, max_core);
        let y_pred:Vec<Array2<f32>> = (0..max_core).map(|_| Array2::<f32>::zeros((max_batch, output_dim))).collect();
        let dW_buffer = self.w.copy_and_fill(0.0);
        let db_buffer = self.b.copy_and_fill(0.0);

        self.train_core(
            layer, epoch,
            dW_buffer, db_buffer, lr,
            z, a, y_pred, false, 
            0,max_core, &clr_format(path),&optimizer,loss_id, batch_size
        );
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
        local_dW: &mut PackedTensor2D,
        local_db: &mut PackedTensor1D,
        act_id:&Vec<usize>,
        act_data:&Vec<f32>,
        dZ: &mut Array2<f32>,
        act_deriv_deriv: &dyn Fn(ArrayView2<f32>, usize, f32) -> Array2<f32>,
    ){
        for l in (0..layer).rev() {
            local_dW.get_mut(l).assign(&a_batch[l].t().dot(dZ));
            local_db.get_mut(l).assign(&dZ.sum_axis(Axis(0)));

            if l == 0 { break }

            let dA = dZ.dot(&w.get(l).t());
            let grad = act_deriv_deriv(z_batch[l - 1].view(), act_id[l - 1], act_data[l-1]);
            *dZ = &dA * &grad;
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
        loss_id: (usize, f32),
        batch_size: usize
    )
    {
        let (loss_id, data_loss) = loss_id;
        let act_id:&Vec<usize> = &self.activations;
        let act_data:&Vec<f32> = &self.data_act;

        let mut best_loss:f32 = 9999.9;
        let mut count_save:usize = 10;

        let (y_data, X_data, total_samples) = if pretrain {
            (&self.y.slice(s![0..N, ..]), &self.X.slice(s![0..N, ..]), N)
        } else {
            (&self.y.view(), &self.X.view(), self.X.nrows())
        };

        let act_deriv_deriv = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, true, data).deriv(z)
        };
        let act_deriv_activate = |z: ArrayView2<f32>, id: usize, data:f32| {
            Activation::from_id(id, false, data).activate(z)
        };

        let loss_func = LossMode::from_id(loss_id, false, data_loss);
        let dz_func = LossMode::from_id(loss_id, true, data_loss);

        let mut optimizer = init_optimizer(&vec![&self.w],&vec![&self.b],optimizer); 
        let chunks = split_range(total_samples, max_core);

        for epoch in 0..epochs{
            let mut epoch_loss = 0.0;

            let results:Vec<_> = chunks.par_iter()
            .zip(a_cache_all_batch.par_iter_mut())
            .zip(z_cache_all_batch.par_iter_mut())
            .zip(y_pred_all_batch.par_iter_mut())
            .map(|(((chunk, a_cache), z_cache), y_pred)| {
                let (start, end) = *chunk;
                // let end1 = if pretrain { N } else { end };
                // let end2 = if pretrain { N } else { end - start };
                let batch_samples = end - start;
                let num_mini_batches = (batch_samples + batch_size - 1) / batch_size;

                let mut total_dW = self.w.copy_and_fill(0.0);
                let mut total_db = self.b.copy_and_fill(0.0);
                let mut total_loss = 0.0;

                for mb_idx in 0..num_mini_batches {
                    let mb_start = mb_idx * batch_size;
                    let mb_end = ((mb_idx + 1) * batch_size).min(batch_samples);
                    let mb_size = mb_end - mb_start;
                    
                    let mut a_batch = a_cache.iter_mut()
                        .map(|a| a.slice_mut(s![mb_start..mb_end, ..]))
                        .collect::<Vec<_>>();
                    let mut z_batch = z_cache.iter_mut()
                        .map(|z| z.slice_mut(s![mb_start..mb_end, ..]))
                        .collect::<Vec<_>>();
                    let mut y_pred_mb = y_pred.slice_mut(s![mb_start..mb_end, ..]);
                    
                    let X_mb = X_data.slice(s![start + mb_start..start + mb_end, ..]);
                    let y_mb = y_data.slice(s![start + mb_start..start + mb_end, ..]);
                    
                    a_batch[0].assign(&X_mb);
                    FNN::forward_pass(layer, &self.w, &self.b,&mut a_batch, &mut z_batch, &mut y_pred_mb,act_id, act_data, &act_deriv_activate);
                    let loss = loss_func.loss(&y_mb, &y_pred_mb.view());
                    total_loss += loss * mb_size as f32;
                    
                    let mut dZ = dz_func.dz(&y_mb, &y_pred_mb.view(), mb_size as f32);
                    
                    let mut local_dW = self.w.copy_and_fill(0.0);
                    let mut local_db = self.b.copy_and_fill(0.0);
                    
                    FNN::backward_pass(
                        layer, &self.w,
                        &mut a_batch, &mut z_batch,
                        &mut local_dW, &mut local_db,
                        act_id, act_data, &mut dZ, &act_deriv_deriv
                    );
                    
                    // Akumulasi gradien
                    for i in 0..layer {
                        total_dW.get_mut(i).add_assign(&local_dW.get(i));
                        total_db.get_mut(i).add_assign(&local_db.get(i));
                    }
                }

                let avg_factor = 1.0 / batch_samples as f32;
                for i in 0..layer {
                    let mut tdw = total_dW.get_mut(i);
                    tdw  *= avg_factor;
                    let mut tdb = total_db.get_mut(i);
                    tdb *= avg_factor;
                }
                
                let avg_loss = total_loss / batch_samples as f32;
                (avg_loss, total_dW, total_db)
                
            }).collect();
            
            dW_buffer.fill_all(0.0);
            db_buffer.fill_all(0.0);

            for (loss, local_dW, local_db) in results{
                epoch_loss += loss;
                for i in 0..layer {
                    dW_buffer.get_mut(i).add_assign(&local_dW.get(i));
                    db_buffer.get_mut(i).add_assign(&local_db.get(i));
                }
            }

            optimizer.run(&mut vec![&mut self.w], &mut vec![&mut self.b], &vec![&dW_buffer], &vec![&db_buffer], lr);

            save_checkpoint_model(epoch_loss, max_core, &mut best_loss, epoch, epochs, pretrain, &mut count_save, &vec![&self.w], &vec![&self.b], act_id, act_data, loss_id, data_loss, &self.labels, path, 1);
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

