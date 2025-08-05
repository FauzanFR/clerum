use ndarray::{Array2, ArrayView2, Axis};
use super::helper::{binary_crossentropy, sign_scalar, weighted_binary_crossentropy};

pub enum LossMode {
    MSE,                  // Loss
    MAE,
    Huber(f32),
    BCE,
    WeightedBCE(f32),
    CrossEntropy,
    MSE_Deriv,            // Derivatives
    MAE_Deriv,
    Huber_Deriv(f32),
    BCE_Deriv,
    WeightedBCE_Deriv(f32),
    CrossEntropy_Deriv,
}

impl LossMode  {
    pub fn from_id(id: usize, deriv: bool, data:f32) -> Self {
        match (id, deriv) {
            (0, false) => LossMode::MSE,
            (1, false) => LossMode::MAE,
            (2, false) => LossMode::Huber(data),
            (3, false) => LossMode::BCE,
            (4, false) => LossMode::WeightedBCE(data),
            (5, false) => LossMode::CrossEntropy,
            (0, true)  => LossMode::MSE_Deriv,
            (1, true)  => LossMode::MAE_Deriv,
            (2, true)  => LossMode::Huber_Deriv(data),
            (3, true)  => LossMode::BCE_Deriv,
            (4, true)  => LossMode::WeightedBCE_Deriv(data),
            (5, true)  => LossMode::CrossEntropy_Deriv,
            _ => panic!("Unknown loss ID: ({}, deriv={})", id, deriv),
        }
    }

    pub fn dz(&self, y: &ArrayView2<f32>, y_pred: &ArrayView2<f32>, n: f32) -> Array2<f32> {
        match *self {
            LossMode::MSE_Deriv => (y_pred - y) * (2.0 / n),
            LossMode::MAE_Deriv => (y_pred - y).mapv_into(|v| v.signum()) / n,
            LossMode::Huber_Deriv(delta) => {
                let error = y_pred - y;
                error.mapv_into(|v| if v.abs() <= delta { v } else { delta * sign_scalar(v) }) / n
            }
            LossMode::BCE_Deriv => (y_pred - y) / n,
            LossMode::WeightedBCE_Deriv(w) => (y_pred - y).mapv_into(|v| v * w) / n,
            LossMode::CrossEntropy_Deriv => (y_pred - y) / n,
            _ => panic!("Cannot call `dz()` on a non-derivative variant"),
        }
    }

    pub fn loss(&self, y: &ArrayView2<f32>, y_pred: &ArrayView2<f32>) -> f32 {
        match *self {
            LossMode::MSE => (y_pred - y).mapv_into(|v| v.powi(2)).mean().unwrap_or(0.0),
            LossMode::MAE => (y_pred - y).mapv_into(|v| v.abs()).mean().unwrap_or(0.0),
            LossMode::Huber(delta) => {
                let error = y_pred - y;
                error.mapv_into(|v| if v.abs() <= delta {
                    0.5 * v.powi(2)
                } else {
                    delta * (v.abs() - 0.5 * delta)
                }).mean().unwrap_or(0.0)
            },
            LossMode::BCE => binary_crossentropy(y, y_pred),
            LossMode::WeightedBCE(w) => weighted_binary_crossentropy(y, y_pred, w),
            LossMode::CrossEntropy => {
                const EPSILON: f32 = 1e-7;
                let clipped = y_pred.mapv(|v| v.clamp(EPSILON, 1.0 - EPSILON));
                let loss_per_sample = - (y * clipped.mapv_into(f32::ln)).sum_axis(Axis(1));
                loss_per_sample.mean().unwrap_or(0.0)
            },
            _ => panic!("Cannot call `loss()` on a derivative variant"),
        }
    }
}

pub enum LossConfig {
    MSE,
    MAE,
    Huber(f32),
    BCE,
    WeightedBCE(f32),
    CrossEntropy,
}

impl LossConfig {
    pub fn loss_id(self) -> (usize, f32) {
        match self {
            LossConfig::MSE => (0,0.0),
            LossConfig::MAE => (1, 0.0),
            LossConfig::Huber(H) => (2, H),
            LossConfig::BCE => (3, 0.0),
            LossConfig::WeightedBCE(W) => (4, W),
            LossConfig::CrossEntropy => (5, 0.0),
        }
    }
}

