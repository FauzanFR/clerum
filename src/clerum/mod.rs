#![allow(non_snake_case, non_camel_case_types)]


pub mod activation;
pub mod file;
pub mod helper;
pub mod loss;
pub mod optimizer;

use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::{clerum::{activation::{Activation, ActivationConfig}, file::{save_checkpoint, Checkpoint}, helper::{clr_format, rand_arr2d, split_range}, loss::{LossConfig, LossMode}, optimizer::{Adam, Momentum, Optimizer, OptimizerConfig, RMSprop}}, file::load_checkpoint};

pub struct FNN{
    X:Array2<f32>,
    y:Array2<f32>,
    w: Vec<Array2<f32>>,
    b: Vec<Array1<f32>>,
    input_dim: Vec<usize>,
    output_dim: Vec<usize>,
    activations: Vec<usize>,
    data_act: Vec<f32>,
    labels: Vec<i32>,

}

impl FNN {
    pub fn init(X:Array2<f32>, y:Array2<f32>) -> Self {
        Self{
            X: X, y: y, w:Vec::new(), b:Vec::new(),
            input_dim:Vec::new(), output_dim:Vec::new(),
            activations:Vec::new(), data_act:Vec::new(),
            labels: Vec::new()
        }
    }

    pub fn from_checkpoint(&mut self, path:&str) {
        let checkpoint = load_checkpoint(path).expect("Failed to load checkpoint");

        self.w = checkpoint.weights;
        self.b = checkpoint.biases;
        self.activations = checkpoint.activation_id;
        self.data_act = checkpoint.data_act;
        self.labels = checkpoint.labels;

        for weight in &self.w {
            let (x, y) = weight.dim();
            self.input_dim.push(x);
            self.output_dim.push(y);
        }
    }

    pub fn add_layer(&mut self, input_dim:usize, output_dim:usize, activation:ActivationConfig) {
        self.w.push(rand_arr2d(input_dim, output_dim));
        self.b.push(Array1::zeros(output_dim));
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

    fn build_backward_buffer_dW(&self) -> Vec<Array2<f32>>{
        let dW: Vec<Array2<f32>> = self.w.iter().map(|w| Array2::zeros(w.raw_dim())).collect();
        dW
    }

    fn build_backward_buffer_db(&self) -> Vec<Array1<f32>>{
        let db: Vec<Array1<f32>> = self.output_dim.iter().map(|&out_dim| Array1::zeros(out_dim)).collect();
        db
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
            let dW_buffer = self.build_backward_buffer_dW();
            let db_buffer = self.build_backward_buffer_db();

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
        let dW_buffer = self.build_backward_buffer_dW();
        let db_buffer = self.build_backward_buffer_db();


        self.train_core(
            layer, epoch,
            dW_buffer, db_buffer, lr,
            z, a, y_pred, false, 
            0,max_core, &clr_format(path),&optimizer,loss_id
        );
    }

    fn train_core(
        &mut self,
        layer:usize,
        epochs:usize,
        mut dW_buffer:Vec<Array2<f32>>,
        mut db_buffer:Vec<Array1<f32>>,
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
        let X_sliced = self.X.slice(s![0..N, ..]);
        let y_sliced = self.y.slice(s![0..N, ..]);
        
        let (loss_id, data_loss) = loss_id;
        let act_id:&Vec<usize> = &self.activations;
        let act_data:&Vec<f32> = &self.data_act;

        let mut best_loss:f32 = 9999.9;
        let mut count_save:usize = 10;

        let (y_all_batch, X_all_batch) = if pretrain {
            (&y_sliced, &X_sliced)
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

        let mut optimizer = match optimizer {
            OptimizerConfig::SGD => Optimizer::SGD,
            OptimizerConfig::Momentum(gamma) => Optimizer::Momentum(Momentum::init(w,b,*gamma)),
            OptimizerConfig::RMSprop(gamma) => Optimizer::RMSprop(RMSprop::init(w,b, *gamma)),
            OptimizerConfig::Adam(b1, b2) => Optimizer::Adam(Adam::init(w,b,*b1, *b2,0)),
        };

        optimizer = optimizer.init(w, b); 


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
                for l in 0..layer {
                    z_batch[l].assign(&a_batch[l].dot(&w[l]));
                    z_batch[l] += &b[l];
                    if l < layer - 1 {
                        a_batch[l + 1].assign(&act_deriv_activate(z_batch[l].view(), act_id[l], act_data[l]));
                    } else {
                        y_pred.assign(&z_batch[l]);
                    }
                }
                let loss = loss_func.loss(&y_batch, &y_pred.view());
                let mut dZ = dz_func.dz(&y_batch, &y_pred.view(), n);

                let mut local_dW = vec![Array2::zeros(w[0].raw_dim()); layer];
                let mut local_db = vec![Array1::zeros(b[0].raw_dim()); layer];

                for l in (0..layer).rev() {
                    local_dW[l] = a_batch[l].t().dot(&dZ);
                    local_db[l] = dZ.sum_axis(Axis(0));

                    if l == 0 { break }

                    let dA = dZ.dot(&w[l].t());
                    let grad = act_deriv_deriv(z_batch[l - 1].view(), act_id[l - 1], act_data[l-1]);
                    dZ = dA * grad

                }


                (loss, local_dW, local_db)
                
            }).collect();

            for i in 0..layer {
                dW_buffer[i].fill(0.0);
                db_buffer[i].fill(0.0);
            }

            for (loss, local_dW, local_db) in results{
                epoch_loss += loss;
                for i in 0..layer {
                    dW_buffer[i] += &local_dW[i];
                    db_buffer[i] += &local_db[i];
                }
            }

            optimizer.run(w, b, &dW_buffer, &db_buffer, lr);

            let avg_loss = epoch_loss / max_core as f32;
            let is_improved = best_loss - avg_loss > 1e-3;
            let is_last_epoch = epoch + 1 == epochs;
            
            if (is_improved && !pretrain) || (is_last_epoch && !pretrain) {

                if is_improved{
                    best_loss = avg_loss;
                    let clr = Checkpoint{
                        epoch,
                        loss:best_loss, 
                        weights:w.to_vec(),
                        biases:b.to_vec(),
                        activation_id:act_id.to_vec(),
                        data_act:act_data.to_vec(),
                        loss_id:loss_id,
                        data_loss:data_loss,
                        labels:self.labels.clone()
                    };

                    if count_save >= 10 || is_last_epoch{
                        println!("Checkpoint saved at epoch {} with loss {:.6}", epoch + 1, avg_loss);
                        count_save = 0;
                        if let Err(e) = save_checkpoint(path, &clr) {
                            eprintln!("Failed to save checkpoint: {:?}", e);
                        }
                    }
                }
            }

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