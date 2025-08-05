use ndarray::{Array1, Array2};


pub fn SGD(weights: &mut Vec<Array2<f32>>, biases:&mut Vec<Array1<f32>>, d_weights:&Vec<Array2<f32>>, d_biases:&Vec<Array1<f32>>, learning_rate:f32){
    for ((w, dw), (b, db)) in weights
    .iter_mut()
    .zip(d_weights)
    .zip(biases.iter_mut().zip(d_biases))
    {
        w.scaled_add(-learning_rate, dw);
        b.scaled_add(-learning_rate, db);
    }
}
pub struct Momentum{
    Velocity_weights: Vec<Array2<f32>>,
    Velocity_biases: Vec<Array1<f32>>,
    gamma:f32
}

impl Momentum{

    pub fn init (weights: &mut Vec<Array2<f32>>, biases:&mut Vec<Array1<f32>>, gamma:f32) -> Self {
        let mut Velocity_weights = Vec::new();
        let mut Velocity_biases = Vec::new();
        for (w, b) in weights
        .iter_mut()
        .zip(biases){
            Velocity_weights.push(Array2::zeros(w.raw_dim()));
            Velocity_biases.push(Array1::zeros(b.raw_dim()));
        }
        Self {
            Velocity_weights,
            Velocity_biases,
            gamma
        }
    }

    fn count_velocity (&mut self, d_weights:&Vec<Array2<f32>>, d_biases:&Vec<Array1<f32>>,learning_rate:f32){
        for ((vw, dw), (vb, db)) in 
        self.Velocity_weights.iter_mut().zip(d_weights.iter())
        .zip(self.Velocity_biases.iter_mut().zip(d_biases.iter())){

            vw.mapv_inplace(|x| x * self.gamma);
            vw.scaled_add(learning_rate, dw);

            vb.mapv_inplace(|x| x * self.gamma);
            vb.scaled_add(learning_rate, db);
        }
    }

    pub fn run (&mut self,weights: &mut Vec<Array2<f32>>, biases:&mut Vec<Array1<f32>>,
        d_weights:&Vec<Array2<f32>>, d_biases:&Vec<Array1<f32>>,
        learning_rate:f32)
        {

            self.count_velocity(d_weights, d_biases, learning_rate);

            for ((w, vw), (b, vb)) in weights
            .iter_mut()
            .zip(self.Velocity_weights.iter())
            .zip(biases.iter_mut().zip(self.Velocity_biases.iter())){

                w.scaled_add(-1.0, &vw);
                b.scaled_add(-1.0, &vb);
        }
    }
}

pub struct RMSprop{
    cache_weights: Vec<Array2<f32>>,
    cache_biases: Vec<Array1<f32>>,
    gamma:f32
}

impl  RMSprop {
    pub fn init (weights: &mut Vec<Array2<f32>>, biases:&mut Vec<Array1<f32>>, gamma:f32) -> Self {
        let mut cache_weights = Vec::new();
        let mut cache_biases = Vec::new();

        for (w, b) in weights
        .iter_mut()
        .zip(biases){
            cache_weights.push(Array2::zeros(w.raw_dim()));
            cache_biases.push(Array1::zeros(b.raw_dim()));
        }
        Self {
            cache_weights,
            cache_biases,
            gamma
        }
    }

    fn count_velocity (&mut self, d_weights:&Vec<Array2<f32>>, d_biases:&Vec<Array1<f32>>){
        for ((vw, dw), (vb, db)) in
        self.cache_weights.iter_mut().zip(d_weights.iter())
        .zip(self.cache_biases.iter_mut().zip(d_biases.iter())){

            for (v, g) in vw.iter_mut().zip(dw.iter()) {
                *v = self.gamma * *v + (1.0 - self.gamma) * g.powi(2);
            }

            for (v, g) in vb.iter_mut().zip(db.iter()) {
                *v = self.gamma * *v + (1.0 - self.gamma) * g.powi(2);
            }


        }
    }
    
    pub fn run (&mut self,weights: &mut Vec<Array2<f32>>, biases:&mut Vec<Array1<f32>>,
        d_weights:&Vec<Array2<f32>>, d_biases:&Vec<Array1<f32>>,
        learning_rate:f32)
        {
            const EPSILON: f32 = 1e-7;
            self.count_velocity(d_weights, d_biases);

            for (((w, vw), (b, vb)), (dw, db)) in
            weights.iter_mut().zip(self.cache_weights.iter())
            .zip(biases.iter_mut().zip(self.cache_biases.iter()))
            .zip(d_weights.iter().zip(d_biases)){

                for ((param, v), grad) in w.iter_mut().zip(vw).zip(dw.iter()){
                    *param -= learning_rate * grad / (v + EPSILON).sqrt();
                }

                for ((param, v), grad) in b.iter_mut().zip(vb).zip(db.iter()){
                    *param -= learning_rate * grad / (v + EPSILON).sqrt();
                }
        }

    }
}

pub struct  Adam{
    momentum_weights: Vec<Array2<f32>>,
    momentum_biases: Vec<Array1<f32>>,
    Velocity_weights: Vec<Array2<f32>>,
    Velocity_biases: Vec<Array1<f32>>,
    beta1:f32,
    beta2:f32,
    iterasi:i32
}

impl Adam {
    pub fn init (weights: &mut Vec<Array2<f32>>, biases:&mut Vec<Array1<f32>>, beta1:f32, beta2:f32, iterasi:i32) -> Self {

        let mut momentum_weights = Vec::new();
        let mut momentum_biases = Vec::new();
        let mut Velocity_weights = Vec::new();
        let mut Velocity_biases = Vec::new();

        for (w, b) in weights
        .iter_mut()
        .zip(biases){
            momentum_weights.push(Array2::zeros(w.raw_dim()));
            momentum_biases.push(Array1::zeros(b.raw_dim()));
            Velocity_weights.push(Array2::zeros(w.raw_dim()));
            Velocity_biases.push(Array1::zeros(b.raw_dim()));
        }

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

    fn count_momentum_velocity (&mut self, d_weights:&Vec<Array2<f32>>, d_biases:&Vec<Array1<f32>>){

        for (((mw, dw), (mb, db)),(vw, vb)) in 
        self.momentum_weights.iter_mut().zip(d_weights.iter())
        .zip(self.momentum_biases.iter_mut().zip(d_biases.iter()))
        .zip(self.Velocity_weights.iter_mut().zip(self.Velocity_biases.iter_mut())){

        for (m, g) in mw.iter_mut().zip(dw.iter()) {
            *m = self.beta1 * *m + (1.0 - self.beta1) * *g;
        }

        for (m, g) in mb.iter_mut().zip(db.iter()) {
            *m = self.beta1 * *m + (1.0 - self.beta1) * *g;
        }

        for (v, g) in vw.iter_mut().zip(dw.iter()) {
            *v = self.beta2 * *v + (1.0 - self.beta2) * g.powi(2);
        }

        for (v, g) in vb.iter_mut().zip(db.iter()) {
            *v = self.beta2 * *v + (1.0 - self.beta2) * g.powi(2);
        }

        }
    }

    pub fn run (&mut self,weights: &mut Vec<Array2<f32>>, biases:&mut Vec<Array1<f32>>,
        d_weights:&Vec<Array2<f32>>, d_biases:&Vec<Array1<f32>>,
        learning_rate:f32)
        {
            const EPSILON: f32 = 1e-7;
            self.iterasi += 1;
            self.count_momentum_velocity(d_weights, d_biases);
            
            let t = self.iterasi as f32;
            let beta1_correction = 1.0 - self.beta1.powf(t);
            let beta2_correction = 1.0 - self.beta2.powf(t);

            for ((((w, mw), vw), (b, mb)), vb) in 
            weights.iter_mut().zip(self.momentum_weights.iter())
            .zip(self.Velocity_weights.iter())
            .zip(biases.iter_mut().zip(self.momentum_biases.iter()))
            .zip(self.Velocity_biases.iter()) {

            for ((param, m), v) in w.iter_mut().zip(mw.iter()).zip(vw.iter()) {
                let m_hat = m / beta1_correction;
                let v_hat = v / beta2_correction;
                *param -= learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
            }

            for ((param, m), v) in b.iter_mut().zip(mb.iter()).zip(vb.iter()) {
                let m_hat = m / beta1_correction;
                let v_hat = v / beta2_correction;
                *param -= learning_rate * m_hat / (v_hat.sqrt() + EPSILON);
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
    pub fn init(self,w: &mut Vec<Array2<f32>>, b: &mut Vec<Array1<f32>>) -> Self {
        match self {
            Optimizer::SGD => Optimizer::SGD,
            Optimizer::Momentum(m) => Optimizer::Momentum(Momentum::init(w, b, m.gamma)),
            Optimizer::RMSprop(r) => Optimizer::RMSprop(RMSprop::init(w, b, r.gamma)),
            Optimizer::Adam(a) => Optimizer::Adam(Adam::init(w, b, a.beta1, a.beta2, 0)),
        }

    }
    pub fn run(
        &mut self,
        w: &mut Vec<Array2<f32>>,
        b: &mut Vec<Array1<f32>>,
        dW: &Vec<Array2<f32>>,
        db: &Vec<Array1<f32>>,
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