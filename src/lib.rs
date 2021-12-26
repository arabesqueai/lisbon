// Copyright (c) 2021 Tony Yang, Arabesque AI
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

#[allow(non_snake_case)]
mod MT19937;
mod linear;
use std::convert::TryInto;

use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;

use crate::linear::{Parameter, Problem};

#[pyfunction]
fn set_verbosity_wrap(_verbose: i64) -> PyResult<()> {
    Ok(())
}

#[allow(non_snake_case)]
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
        let slices = dense_to_slices(&x);
        let target = y.as_slice().unwrap();
        let l = x.shape()[0];
        let n = x.shape()[1];
        let b = if bias > 0.0 { bias } else { 0.0 };
        let prob = Problem::new(l, n, target, slices, b);
        let param = Parameter {
            solver_type: solver_type.try_into().unwrap(),
            eps,
            C,
            max_iter,
            random_seed,
        };
        let (n_iter, w) = linear::train(&prob, param);
        let weight = Array2::from_shape_vec((1, w.len()), w)
            .unwrap()
            .into_pyarray(py);
        let n_iter = vec![n_iter].into_pyarray(py);
        Ok((weight, n_iter))
    }
    Ok(())
}

fn dense_to_slices<'a>(arr: &'a PyReadonlyArray2<f64>) -> Vec<&'a [f64]> {
    let n = arr.shape()[1];
    arr.as_slice().unwrap().chunks(n).collect()
}
