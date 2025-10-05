use std::{borrow::Cow, sync::{Mutex, LazyLock}};
use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Normal, Distribution};

static SEEDED_RNG: LazyLock<Mutex<ChaCha8Rng>> = LazyLock::new(|| {
    Mutex::new(ChaCha8Rng::seed_from_u64(42))  // Default seed
});

pub fn set_global_seed(seed: u64) {
    let mut rng = SEEDED_RNG.lock().unwrap();
    *rng = ChaCha8Rng::seed_from_u64(seed);
}

pub fn rand_arr1d(x:usize) -> Array1<f32> {
    let normal:Normal<f32> = Normal::new(0.0, 1.0).unwrap();
    let mut rng = SEEDED_RNG.lock().unwrap();

    Array1::from_shape_fn(x, |_| normal.sample(&mut rng))
}

pub fn rand_arr2d(x:usize, y:usize) -> Array2<f32> {
    let normal:Normal<f32> = Normal::new(0.0, 1.0).unwrap();
    let mut rng = SEEDED_RNG.lock().unwrap();
    Array2::from_shape_fn((x,y), |_| normal.sample(&mut rng))
}

pub fn rand_arr3d(x:usize, y:usize, z:usize) -> Array3<f32> {
    let normal:Normal<f32> = Normal::new(0.0, 1.0).unwrap();
    let mut rng = SEEDED_RNG.lock().unwrap();
    Array3::from_shape_fn((x,y,z), |_| normal.sample(&mut rng))
}

pub fn rand_arr1d_seeded(x: usize, seed: u64) -> Array1<f32> {
    let normal: Normal<f32> = Normal::new(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array1::from_shape_fn(x, |_| normal.sample(&mut rng))
}

pub fn rand_arr2d_seeded(x: usize, y: usize, seed: u64) -> Array2<f32> {
    let normal: Normal<f32> = Normal::new(0.0, 1.0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((x, y), |_| normal.sample(&mut rng))
}

pub fn _rand_vec2d(x:usize, y:usize) -> Vec<Vec<f32>> {
    (0..y).map(|_| _rand_vec1d(x)).collect()
}

pub fn _rand_vec1d(x:usize) -> Vec<f32> {
    let normal:Normal<f32> = Normal::new(0.0, 1.0).unwrap();
    let mut rng = SEEDED_RNG.lock().unwrap();
    (0..x).map(|_| normal.sample(&mut rng)).collect()
}

pub fn _vec_to_array1d(x:Vec<f32>)->Array1<f32>{
    Array1::from(x)
}

pub fn _sign_array(arr: &Array2<f32>) -> Array2<f32> {
    arr.mapv(sign_scalar)
}

pub fn sign_scalar(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

pub fn binary_crossentropy(y_true: &ArrayView2<f32>, y_pred: &ArrayView2<f32>) -> f32 {
    const EPSILON: f32 = 1e-7;
    let clipped = y_pred.mapv(|v| v.clamp(EPSILON, 1.0 - EPSILON));

    let loss = y_true * &clipped.mapv(f32::ln) + (1.0 - y_true) * &(1.0 - &clipped).mapv(f32::ln);
    -loss.mean().unwrap_or(0.0)
}

pub fn _binary_crossentropy_array(y_true: &Array2<f32>, y_pred: &Array2<f32>) -> Array2<f32> {
    const EPSILON: f32 = 1e-7;
    let clipped = y_pred.mapv(|v| v.clamp(EPSILON, 1.0 - EPSILON));
    y_true * &clipped.mapv(f32::ln) + (1.0 - y_true) * &(1.0 - &clipped).mapv(f32::ln)
}


pub fn weighted_binary_crossentropy(y_true: &ArrayView2<f32>, y_pred: &ArrayView2<f32>, pos_weight: f32) -> f32 {
    const EPSILON: f32 = 1e-7;
    let clipped = y_pred.mapv(|v| v.clamp(EPSILON, 1.0 - EPSILON));
    let part1 = y_true * &clipped.mapv(f32::ln) * pos_weight;
    let part2 = (1.0 - y_true) * (1.0 - &clipped).mapv(f32::ln);
    let loss = -(part1 + part2);
    let mean_loss = loss.mean().unwrap_or(0.0);

    mean_loss
}

pub fn _weighted_binary_crossentropy_array(
    y_true: &Array2<f32>,
    y_pred: &Array2<f32>,
    pos_weight: f32,
) -> Array2<f32> {
    const EPSILON: f32 = 1e-7;
    let clipped = y_pred.mapv(|v| v.clamp(EPSILON, 1.0 - EPSILON));

    let part1 = y_true * &clipped.mapv(f32::ln) * pos_weight;
    let part2 = (1.0 - y_true) * (1.0 - &clipped).mapv(f32::ln);

    part1 + part2 // belum dikalikan -1
}

pub fn _batchify_input_layers(
    layers: &Vec<Array2<f32>>,
    batch_count: usize,
) -> Vec<Vec<Array2<f32>>> {
    let n_samples = layers[0].nrows();
    let batch_size = n_samples / batch_count;
    let mut result: Vec<Vec<Array2<f32>>> = Vec::with_capacity(batch_count);

    for b in 0..batch_count {
        let start = b * batch_size;
        let end = if b == batch_count - 1 {
            n_samples
        } else {
            (b + 1) * batch_size
        };

        let mut batch_layers = Vec::with_capacity(layers.len());
        for layer in layers {
            let sliced = layer.slice(s![start..end, ..]).to_owned();
            batch_layers.push(sliced);
        }

        result.push(batch_layers);
    }

    result
}

pub fn _split_array2_to_batches(data: &Array2<f32>, batch_count: usize) -> Vec<Array2<f32>> {
    let total_rows = data.nrows();
    let batch_size = total_rows / batch_count;
    let mut batches = Vec::with_capacity(batch_count);

    for i in 0..batch_count {
        let start = i * batch_size;
        let end = if i == batch_count - 1 {
            total_rows // batch terakhir ambil sisanya
        } else {
            (i + 1) * batch_size
        };
        let batch = data.slice(s![start..end, ..]).to_owned();
        batches.push(batch);
    }

    batches
}

pub fn split_range(max_batch: usize, chunk_count: usize) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(chunk_count);
    let mut start = 0;

    for i in 0..chunk_count {
        // Bagi rata + distribusi sisa ke awal
        let extra = if i < max_batch % chunk_count { 1 } else { 0 };
        let end = start + (max_batch / chunk_count) + extra;
        result.push((start, end));
        start = end;
    }

    result
}

pub fn clr_format<'a>(name: &'a str) -> Cow<'a, str> {
    if name.ends_with(".clr") {
        Cow::Borrowed(name)
    } else {
        Cow::Owned(format!("{}.clr", name))
    }
}