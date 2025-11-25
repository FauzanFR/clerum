use TensorClerum::{tensor1::PackedTensor1D, tensor2::PackedTensor2D};

pub fn SGD(
    weights: &mut [&mut PackedTensor2D],
    biases: &mut [&mut PackedTensor1D],
    d_weights: &[&PackedTensor2D],
    d_biases: &[&PackedTensor1D],
    learning_rate: f32,
) {
    for (w, dw) in weights.iter_mut().zip(d_weights) {
        for i in 0..w.len() {
            w.get_mut(i).scaled_add(learning_rate, &dw.get(i));
        }
    }
    for (b, db) in biases.iter_mut().zip(d_biases) {
        for i in 0..b.len() {
            b.get_mut(i).scaled_add(learning_rate, &db.get(i));
        }
    }
}

pub struct Momentum {
    velocity_weights: Vec<PackedTensor2D>,
    velocity_biases: Vec<PackedTensor1D>,
    gamma: f32,
}

impl Momentum {
    pub fn init(weights: &[&PackedTensor2D], biases: &[&PackedTensor1D], gamma: f32) -> Self {
        let velocity_weights: Vec<PackedTensor2D> = weights.iter().map(|f| f.copy_and_fill(0.0)).collect();
        let velocity_biases: Vec<PackedTensor1D> = biases.iter().map(|f| f.copy_and_fill(0.0)).collect();

        Self {
            velocity_weights,
            velocity_biases,
            gamma,
        }
    }

    fn count_velocity(&mut self, d_weights: &[&PackedTensor2D], d_biases: &[&PackedTensor1D], learning_rate: f32) {
        for (vw, dw) in self.velocity_weights.iter_mut().zip(d_weights) {
            for i in 0..dw.len() {
                let mut vw_layer = vw.get_mut(i);
                vw_layer *= self.gamma;
                vw_layer.scaled_add(learning_rate, &dw.get(i));
            }
        }

        for (vb, db) in self.velocity_biases.iter_mut().zip(d_biases) {
            for i in 0..db.len() {
                let mut vb_layer = vb.get_mut(i);
                vb_layer *= self.gamma;
                vb_layer.scaled_add(learning_rate, &db.get(i));
            }
        }
    }

    pub fn run(
        &mut self,
        weights: &mut [&mut PackedTensor2D],
        biases: &mut [&mut PackedTensor1D],
        d_weights: &[&PackedTensor2D],
        d_biases: &[&PackedTensor1D],
        learning_rate: f32,
    ) {
        self.count_velocity(d_weights, d_biases, learning_rate);

        for (w, vw) in weights.iter_mut().zip(&self.velocity_weights) {
            for i in 0..w.len() {
                w.get_mut(i).scaled_add(-1.0, &vw.get(i));
            }
        }

        for (b, vb) in biases.iter_mut().zip(&self.velocity_biases) {
            for i in 0..b.len() {
                b.get_mut(i).scaled_add(-1.0, &vb.get(i));
            }
        }
    }
}

pub struct RMSprop {
    cache_weights: Vec<PackedTensor2D>,
    cache_biases: Vec<PackedTensor1D>,
    rho: f32,
}

impl RMSprop {
    pub fn init(weights: &[&PackedTensor2D], biases: &[&PackedTensor1D], rho: f32) -> Self {
        let cache_weights: Vec<PackedTensor2D> = weights.iter().map(|f| f.copy_and_fill(0.0)).collect();
        let cache_biases: Vec<PackedTensor1D> = biases.iter().map(|f| f.copy_and_fill(0.0)).collect();

        Self {
            cache_weights,
            cache_biases,
            rho,
        }
    }

    fn count_velocity(&mut self, d_weights: &[&PackedTensor2D], d_biases: &[&PackedTensor1D]) {
        for (cw, dw) in self.cache_weights.iter_mut().zip(d_weights) {
            for i in 0..dw.len() {
                let mut cw_layer = cw.get_mut(i);
                let dw_layer = dw.get(i);

                cw_layer.zip_mut_with(&dw_layer, |v, g| {
                    *v = self.rho * *v + (1.0 - self.rho) * g.powi(2);
                });
            }
        }

        for (cb, db) in self.cache_biases.iter_mut().zip(d_biases) {
            for i in 0..db.len() {
                let mut cb_layer = cb.get_mut(i);
                let db_layer = db.get(i);

                cb_layer.zip_mut_with(&db_layer, |v, g| {
                    *v = self.rho * *v + (1.0 - self.rho) * g.powi(2);
                });
            }
        }
    }

    pub fn run(
        &mut self,
        weights: &mut [&mut PackedTensor2D],
        biases: &mut [&mut PackedTensor1D],
        d_weights: &[&PackedTensor2D],
        d_biases: &[&PackedTensor1D],
        learning_rate: f32,
    ) {
        const EPSILON: f32 = 1e-7;
        self.count_velocity(d_weights, d_biases);

        for ((w, cw), dw) in weights.iter_mut().zip(&self.cache_weights).zip(d_weights) {
            for i in 0..w.len() {
                let w_layer = w.get_mut(i);
                let cw_layer = cw.get(i);
                let dw_layer = dw.get(i);

                ndarray::Zip::from(w_layer).and(cw_layer).and(dw_layer)
                    .for_each(|param, &c, &grad| {
                        *param -= learning_rate * grad / (c + EPSILON).sqrt();
                    });
            }
        }

        for ((b, cb), db) in biases.iter_mut().zip(&self.cache_biases).zip(d_biases) {
            for i in 0..b.len() {
                let b_layer = b.get_mut(i);
                let cb_layer = cb.get(i);
                let db_layer = db.get(i);

                ndarray::Zip::from(b_layer).and(cb_layer).and(db_layer)
                    .for_each(|param, &c, &grad| {
                        *param -= learning_rate * grad / (c + EPSILON).sqrt();
                    });
            }
        }
    }
}

pub struct Adam {
    momentum_weights: Vec<PackedTensor2D>,
    momentum_biases: Vec<PackedTensor1D>,
    velocity_weights: Vec<PackedTensor2D>,
    velocity_biases: Vec<PackedTensor1D>,
    beta1: f32,
    beta2: f32,
    iterasi: i32,
}

impl Adam {
    pub fn init(weights: &[&PackedTensor2D], biases: &[&PackedTensor1D], beta1: f32, beta2: f32, iterasi: i32) -> Self {
        let momentum_weights: Vec<PackedTensor2D> = weights.iter().map(|f| f.copy_and_fill(0.0)).collect();
        let momentum_biases: Vec<PackedTensor1D> = biases.iter().map(|f| f.copy_and_fill(0.0)).collect();
        let velocity_weights: Vec<PackedTensor2D> = weights.iter().map(|f| f.copy_and_fill(0.0)).collect();
        let velocity_biases: Vec<PackedTensor1D> = biases.iter().map(|f| f.copy_and_fill(0.0)).collect();

        Self {
            momentum_weights,
            momentum_biases,
            velocity_weights,
            velocity_biases,
            beta1,
            beta2,
            iterasi,
        }
    }

    fn count_momentum_velocity(&mut self, d_weights: &[&PackedTensor2D], d_biases: &[&PackedTensor1D]) {
        for ((mw, vw), dw) in self.momentum_weights.iter_mut().zip(&mut self.velocity_weights).zip(d_weights) {
            for i in 0..dw.len() {
                let mut mw_layer = mw.get_mut(i);
                let mut vw_layer = vw.get_mut(i);
                let dw_layer = dw.get(i);

                mw_layer.zip_mut_with(&dw_layer, |m, g| {
                    *m = self.beta1 * *m + (1.0 - self.beta1) * *g;
                });

                vw_layer.zip_mut_with(&dw_layer, |v, g| {
                    *v = self.beta2 * *v + (1.0 - self.beta2) * g.powi(2);
                });
            }
        }

        for ((mb, vb), db) in self.momentum_biases.iter_mut().zip(&mut self.velocity_biases).zip(d_biases) {
            for i in 0..db.len() {
                let mut mb_layer = mb.get_mut(i);
                let mut vb_layer = vb.get_mut(i);
                let db_layer = db.get(i);

                mb_layer.zip_mut_with(&db_layer, |m, g| {
                    *m = self.beta1 * *m + (1.0 - self.beta1) * *g;
                });

                vb_layer.zip_mut_with(&db_layer, |v, g| {
                    *v = self.beta2 * *v + (1.0 - self.beta2) * g.powi(2);
                });
            }
        }
    }

    pub fn run(
        &mut self,
        weights: &mut [&mut PackedTensor2D],
        biases: &mut [&mut PackedTensor1D],
        d_weights: &[&PackedTensor2D],
        d_biases: &[&PackedTensor1D],
        learning_rate: f32,
    ) {
        const EPSILON: f32 = 1e-7;
        self.iterasi += 1;
        self.count_momentum_velocity(d_weights, d_biases);

        let bias_correction1 = 1.0 - self.beta1.powi(self.iterasi);
        let bias_correction2 = 1.0 - self.beta2.powi(self.iterasi);

        for ((w, mw), vw) in weights.iter_mut().zip(&self.momentum_weights).zip(&self.velocity_weights) {
            for i in 0..w.len() {
                let w_layer = w.get_mut(i);
                let mw_layer = mw.get(i);
                let vw_layer = vw.get(i);

                ndarray::Zip::from(w_layer).and(mw_layer).and(vw_layer).for_each(
                    |param, &m, &v| {
                        let m_hat = m / bias_correction1;
                        let v_hat = v / bias_correction2;
                        *param -= learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
                    },
                );
            }
        }

        for ((b, mb), vb) in biases.iter_mut().zip(&self.momentum_biases).zip(&self.velocity_biases) {
            for i in 0..b.len() {
                let b_layer = b.get_mut(i);
                let mb_layer = mb.get(i);
                let vb_layer = vb.get(i);

                ndarray::Zip::from(b_layer).and(mb_layer).and(vb_layer).for_each(
                    |param, &m, &v| {
                        let m_hat = m / bias_correction1;
                        let v_hat = v / bias_correction2;
                        *param -= learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
                    },
                );
            }
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
    pub fn init(self, w: &[&PackedTensor2D], b: &[&PackedTensor1D]) -> Self {
        match self {
            Optimizer::SGD => Optimizer::SGD,
            Optimizer::Momentum(m) => Optimizer::Momentum(Momentum::init(w, b, m.gamma)),
            Optimizer::RMSprop(r) => Optimizer::RMSprop(RMSprop::init(w, b, r.rho)),
            Optimizer::Adam(a) => Optimizer::Adam(Adam::init(w, b, a.beta1, a.beta2, 0)),
        }
    }

    pub fn run(
        &mut self,
        w: &mut [&mut PackedTensor2D],
        b: &mut [&mut PackedTensor1D],
        dW: &[&PackedTensor2D],
        db: &[&PackedTensor1D],
        lr: f32,
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

pub fn init_optimizer(w: &[&PackedTensor2D], b: &[&PackedTensor1D], config: &OptimizerConfig) -> Optimizer {
    match config {
        OptimizerConfig::SGD => Optimizer::SGD.init(w, b),
        OptimizerConfig::Momentum(gamma) => Optimizer::Momentum(Momentum::init(w,b,*gamma)),
        OptimizerConfig::RMSprop(gamma) => Optimizer::RMSprop(RMSprop::init(w,b, *gamma)),
        OptimizerConfig::Adam(b1, b2) => Optimizer::Adam(Adam::init(w,b,*b1, *b2,0))
    }
}
