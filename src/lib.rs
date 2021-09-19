mod MT19937;
mod linear;
use std::convert::TryInto;

use ndarray::Array2;
use ndarray::Axis;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::linear::{Parameter, Problem};

#[pyfunction]
fn set_verbosity_wrap(verbose: i64) -> PyResult<()> {
    Ok(())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn lisbon(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_verbosity_wrap, m)?)?;
    #[pyfn(m)]
    fn train_wrap<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        _is_sparse: bool,
        solver_type: usize,
        eps: f64,
        bias: f64,
        C: f64,
        _class_weight: PyReadonlyArray1<'py, f64>,
        max_iter: usize,
        random_seed: u32,
        _epsilon: f64,
        _sample_weight: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<usize>)> {
        let sparse = dense_to_sparse(&x);
        let target = y.to_vec().unwrap();
        let l = x.shape()[0];
        let n = x.shape()[1] + 1;
        let prob = Problem::new(l, n, target, sparse, bias);
        let param = Parameter {
            solver_type: solver_type.try_into().unwrap(),
            eps,
            C,
            nr_weight: 0,
            weight_label: 0,
            weight: 0.0,
            p: 0.0,
            nu: 0.0,
            init_sol: 0.0,
            regularize_bias: 0,
            max_iter,
            random_seed,
        };
        let (model, n_iter) = linear::train(&prob, param);
        let weight = Array2::from_shape_vec((1, model.w.len()), model.w)
            .unwrap()
            .into_pyarray(py);
        let n_iter = vec![n_iter].into_pyarray(py);
        Ok((weight, n_iter))
    }
    Ok(())
}

fn dense_to_sparse(arr: &PyReadonlyArray2<f64>) -> Vec<Vec<f64>> {
    arr.as_array()
        .axis_iter(Axis(0))
        .map(|row| {
            let mut ret = row.to_vec();
            ret.push(1.0);
            ret
        })
        .collect()
}
