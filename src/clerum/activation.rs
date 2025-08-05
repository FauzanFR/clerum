use ndarray::{Array1, Array2, ArrayView2};

pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f32),
    ELU(f32),
    Softplus,
    Linear,
    Softmax,
    ReLuDeriv,
    SigmoidDeriv,
    TanhDeriv,
    LeakyReLuDeriv(f32),
    EluDeriv(f32),
    SoftplusDeriv,
    LinearDeriv,
    SoftmaxDeriv,
}

impl Activation {
    pub fn activate(&self, x:ArrayView2<f32>) -> Array2<f32> {
        match *self {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(f32::tanh),
            Activation::LeakyReLU(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * v }),
            Activation::ELU(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }),
            Activation::Softplus => x.mapv(|v| (1.0 + v.exp()).ln()),
            Activation::Linear => x.to_owned(),
            Activation::Softmax =>{
                let mut result = Array2::zeros(x.raw_dim());
                for (mut out_row, in_row) in result.rows_mut().into_iter().zip(x.rows()) {
                    let max = in_row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp: Array1<f32> = in_row.mapv(|v| (v - max).exp());
                    let sum = exp.sum();
                    out_row.assign(&(exp / sum));
                }
                result
            }

            _ => panic!("Cannot call `activate()` on a derivative variant"),
        }
    }

    pub fn deriv(&self, x: ArrayView2<f32>) -> Array2<f32> {
        match *self {
            Activation::ReLuDeriv => x.mapv(|v| if v > 0.0 {1.0} else {0.0}),
            Activation::SigmoidDeriv => {let s = Activation::Sigmoid.activate(x); s.mapv(|v| v*(1.0-v))},
            Activation::TanhDeriv => {let t = Activation::Tanh.activate(x); t.mapv(|v| 1.0-v.powi(2))},
            Activation::LeakyReLuDeriv(alpha) => x.mapv(|v| if v > 0.0 {1.0} else {alpha}),
            Activation::EluDeriv(alpha) => x.mapv(|v| if v > 0.0 {1.0} else {alpha * v.exp()}),
            Activation::SoftplusDeriv => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::LinearDeriv => x.mapv(|_| 1.0),
            Activation::SoftmaxDeriv => {
                let s = Activation::Softmax.activate(x);
                s.mapv(|v| v * (1.0 - v))
            }

            _ => panic!("Cannot call `deriv()` on a derivative variant"),
        }
    }

    pub fn from_id(id: usize, deriv: bool, data:f32) -> Self {
        match (deriv, id) {
            (false, 0) => Activation::ReLU,
            (false, 1) => Activation::Sigmoid,
            (false, 2) => Activation::Tanh,
            (false, 3) => Activation::LeakyReLU(data),
            (false, 4) => Activation::ELU(data),
            (false, 5) => Activation::Softplus,
            (false, 6) => Activation::Linear,
            (false, 7) => Activation::Softmax,

            (true, 0) => Activation::ReLuDeriv,
            (true, 1) => Activation::SigmoidDeriv,
            (true, 2) => Activation::TanhDeriv,
            (true, 3) => Activation::LeakyReLuDeriv(data),
            (true, 4) => Activation::EluDeriv(data),
            (true, 5) => Activation::SoftplusDeriv,
            (true, 6) => Activation::LinearDeriv,
            (true, 7) => Activation::SoftmaxDeriv,

            _ => panic!("Unknown activation id ({}, {} {})", id, deriv, data),
        }
    }
}

pub enum ActivationConfig {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f32),
    ELU(f32),
    Softplus,
    Linear,
    Softmax
}

impl ActivationConfig {
    pub fn activation_id(self) -> (usize, f32) {
        match self {
            ActivationConfig::ReLU => (0, 0.0),
            ActivationConfig::Sigmoid => (1, 0.0),
            ActivationConfig::Tanh => (2, 0.0),
            ActivationConfig::LeakyReLU(L) => (3, L),
            ActivationConfig::ELU(E) => (4, E),
            ActivationConfig::Softplus => (5, 0.0),
            ActivationConfig::Linear => (6, 0.0),
            ActivationConfig::Softmax => (7, 0.0),
        }
    }
}