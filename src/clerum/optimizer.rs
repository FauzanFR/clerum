use TensorClerum::{tensor1::PackedTensor1D, tensor2::PackedTensor2D};

pub fn SGD(weights: &mut PackedTensor2D, biases:&mut PackedTensor1D, d_weights:&PackedTensor2D, d_biases:&PackedTensor1D, learning_rate:f32){
    
    for i in 0..weights.len(){
        weights.get_mut(i).scaled_add(learning_rate, &d_weights.get(i));
        biases.get_mut(i).scaled_add(learning_rate, &d_biases.get(i));
    }
}
pub struct Momentum{
    Velocity_weights: PackedTensor2D,
    Velocity_biases: PackedTensor1D,
    gamma:f32
}

impl Momentum{

    pub fn init (weights: &PackedTensor2D, biases:&PackedTensor1D, gamma:f32) -> Self {
        let Velocity_weights = weights.copy_and_fill(0.0);
        let Velocity_biases = biases.copy_and_fill(0.0);

        Self {
            Velocity_weights,
            Velocity_biases,
            gamma
        }
    }

    fn count_velocity (&mut self, d_weights:&PackedTensor2D, d_biases:&PackedTensor1D,learning_rate:f32){

        for i in 0..d_biases.len(){
            let mut vw = self.Velocity_weights.get_mut(i);
            vw *= self.gamma;
            vw.scaled_add(learning_rate, &d_weights.get(i));

            let mut vb = self.Velocity_biases.get_mut(i);
            vb *= self.gamma;
            vb.scaled_add(learning_rate, &d_biases.get(i));
        }
    }

    pub fn run (&mut self,weights: &mut PackedTensor2D, biases:&mut PackedTensor1D,
        d_weights:&PackedTensor2D, d_biases:&PackedTensor1D,
        learning_rate:f32)
        {
            self.count_velocity(d_weights, d_biases, learning_rate);

            for i in 0..weights.len(){
                weights.get_mut(i).scaled_add(-1.0, &self.Velocity_weights.get(i));
                biases.get_mut(i).scaled_add(-1.0, &self.Velocity_biases.get(i));
            }
    }
}

pub struct RMSprop{
    cache_weights: PackedTensor2D,
    cache_biases: PackedTensor1D,
    gamma:f32
}

impl  RMSprop {
    pub fn init (weights: &PackedTensor2D, biases:&PackedTensor1D, gamma:f32) -> Self {
        let cache_weights = weights.copy_and_fill(0.0);
        let cache_biases = biases.copy_and_fill(0.0);

        Self {
            cache_weights,
            cache_biases,
            gamma
        }
    }

    fn count_velocity (&mut self, d_weights:&PackedTensor2D, d_biases:&PackedTensor1D){

        for i in 0..d_biases.len(){
            let mut vw = self.cache_weights.get_mut(i);
            let dw = d_weights.get(i);
            vw.zip_mut_with(&dw, |v, g| {
                *v = self.gamma * *v + (1.0 - self.gamma) * g.powi(2);
            });

            let mut vb = self.cache_biases.get_mut(i);
            let db = d_biases.get(i);
            vb.zip_mut_with(&db, |v, g| {
                *v = self.gamma * *v + (1.0 - self.gamma) * g.powi(2);
            });
        }
    }
    
    pub fn run (&mut self,weights: &mut PackedTensor2D, biases:&mut PackedTensor1D,
        d_weights:&PackedTensor2D, d_biases:&PackedTensor1D,
        learning_rate:f32)
        {
            const EPSILON: f32 = 1e-7;
            self.count_velocity(d_weights, d_biases);

            for i in 0..weights.len(){
                let mut w  = weights.get_mut(i);
                let vw     = self.cache_weights.get(i);
                let dw     = d_weights.get(i);

                ndarray::Zip::from(&mut w)
                    .and(&vw)
                    .and(&dw)
                    .for_each(|param, &v, &grad| {
                        *param -= learning_rate * grad / (v + EPSILON).sqrt();
                    });

                let mut b  = biases.get_mut(i);
                let vb     = self.cache_biases.get(i);
                let db     = d_biases.get(i);

                // in-place elementwise update untuk biases
                ndarray::Zip::from(&mut b)
                    .and(&vb)
                    .and(&db)
                    .for_each(|param, &v, &grad| {
                        *param -= learning_rate * grad / (v + EPSILON).sqrt();
                    });
            }

        }
}

pub struct  Adam{
    momentum_weights: PackedTensor2D,
    momentum_biases: PackedTensor1D,
    Velocity_weights: PackedTensor2D,
    Velocity_biases: PackedTensor1D,
    beta1:f32,
    beta2:f32,
    iterasi:i32
}

impl Adam {
    pub fn init (weights: &PackedTensor2D, biases:&PackedTensor1D, beta1:f32, beta2:f32, iterasi:i32) -> Self {

        let momentum_weights = weights.copy_and_fill(0.0);
        let momentum_biases = biases.copy_and_fill(0.0);
        let Velocity_weights = weights.copy_and_fill(0.0);
        let Velocity_biases = biases.copy_and_fill(0.0);

        Self{
            momentum_weights,
            momentum_biases,
            Velocity_weights,
            Velocity_biases,
            beta1,
            beta2,
            iterasi
        }
    }

    fn count_momentum_velocity (&mut self, d_weights:&PackedTensor2D, d_biases:&PackedTensor1D){

        for i in 0..d_weights.len() {
            let mut mw = self.momentum_weights.get_mut(i);
            let mut mb = self.momentum_biases.get_mut(i);
            let mut vw = self.Velocity_weights.get_mut(i);
            let mut vb = self.Velocity_biases.get_mut(i);
            let dw = d_weights.get(i);
            let db = d_biases.get(i);

            mw.zip_mut_with(&dw, |m, g|{
                *m = self.beta1 * *m + (1.0 - self.beta1) * *g
            });

            mb.zip_mut_with(&db, |m, g|{
                *m = self.beta1 * *m + (1.0 - self.beta1) * *g
            });

            vw.zip_mut_with(&dw, |v, g|{
                *v = self.beta2 * *v + (1.0 - self.beta2) * g.powi(2)
            });

            vb.zip_mut_with(&db, |v, g|{
                *v = self.beta2 * *v + (1.0 - self.beta2) * g.powi(2)
            });
        }
    }

    pub fn run (&mut self,weights: &mut PackedTensor2D, biases:&mut PackedTensor1D,
        d_weights:&PackedTensor2D, d_biases:&PackedTensor1D,
        learning_rate:f32)
        {
            const EPSILON: f32 = 1e-7;
            self.iterasi += 1;
            self.count_momentum_velocity(d_weights, d_biases);
            
            let t = self.iterasi as f32;
            let bias_correction1 = 1.0 - self.beta1.powf(t);
            let bias_correction2 = 1.0 - self.beta2.powf(t);

            for i in 0..weights.len(){
                let mw = self.momentum_weights.get(i);
                let mb = self.momentum_biases.get(i);
                let vw = self.Velocity_weights.get(i);
                let vb = self.Velocity_biases.get(i);
                let mut w = weights.get_mut(i);
                let mut b = biases.get_mut(i);

                ndarray::Zip::from(&mut w).and(mw).and(vw).for_each(
                    |param, &m, &v| {
                        let m_hat = m / bias_correction1;
                        let v_hat = v / bias_correction2;
                        *param -= learning_rate *m_hat / (v_hat.sqrt() + EPSILON) 
                    }
                );

                ndarray::Zip::from(&mut b).and(mb).and(vb).for_each(
                    |param, &m, &v| {
                        let m_hat = m / bias_correction1;
                        let v_hat = v / bias_correction2;
                        *param -= learning_rate *m_hat / (v_hat.sqrt() + EPSILON) 
                    }
                );
            }
    }
}

pub enum Optimizer {
    SGD,
    Momentum(Momentum),
    RMSprop(RMSprop),
    Adam(Adam),
}

impl Optimizer {
    pub fn init(self,w: &PackedTensor2D, b: &PackedTensor1D) -> Self {
        match self {
            Optimizer::SGD => Optimizer::SGD,
            Optimizer::Momentum(m) => Optimizer::Momentum(Momentum::init(w, b, m.gamma)),
            Optimizer::RMSprop(r) => Optimizer::RMSprop(RMSprop::init(w, b, r.gamma)),
            Optimizer::Adam(a) => Optimizer::Adam(Adam::init(w, b, a.beta1, a.beta2, 0)),
        }

    }
    pub fn run(
        &mut self,
        w: &mut PackedTensor2D,
        b: &mut PackedTensor1D,
        dW: &PackedTensor2D,
        db: &PackedTensor1D,
        lr: f32
    ) {
        match self {
            Optimizer::SGD => SGD(w, b, dW, db, lr),
            Optimizer::Momentum(opt) => opt.run(w, b, dW, db, lr),
            Optimizer::RMSprop(opt) => opt.run(w, b, dW, db, lr),
            Optimizer::Adam(opt) => opt.run(w, b, dW, db, lr),
        }
    }
}

pub enum OptimizerConfig {
    SGD,
    Momentum(f32),
    RMSprop(f32),
    Adam(f32, f32),
}