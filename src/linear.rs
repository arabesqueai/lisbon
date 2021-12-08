// Copyright (c) 2021 Tony Yang, Arabesque AI
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

use crate::MT19937::MT19937;
use std::convert::TryFrom;

pub struct Problem<'a> {
    l: usize,          // number of training data
    n: usize, // number of features (including the bias feature if bias >= 0)
    y: &'a [f64], // reference to an array of target values
    x: Vec<&'a [f64]>, // array of references to training vectors
    bias: f64, // < 0 if no bias term
}

impl<'a> Problem<'a> {
    pub fn new(
        l: usize,
        n: usize,
        y: &'a [f64],
        x: Vec<&'a [f64]>,
        bias: f64,
    ) -> Self {
        Self { l, n, y, x, bias }
    }
}

#[allow(non_camel_case_types)]
pub enum SolverType {
    L2R_L1LOSS_SVC_DUAL = 3, // L2-regularized L1-loss support vector classification (dual)
}

impl TryFrom<usize> for SolverType {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            3 => Ok(Self::L2R_L1LOSS_SVC_DUAL),
            _ => Err("Solver type does not exist or not implemented"),
        }
    }
}

/// nr_weight, weight_label, and weight are used to change the penalty
/// for some classes (If the weight for a class is not changed, it is
/// set to 1). This is useful for training classifier using unbalanced
/// input data or with asymmetric misclassification cost.
#[allow(non_snake_case)]
pub struct Parameter {
    pub solver_type: SolverType,
    pub eps: f64, // stopping tolerance
    pub C: f64,   // cost of constraints violation
    pub max_iter: usize,
    pub random_seed: u32,
}

pub struct BLASOperator;

impl BLASOperator {
    #[inline]
    pub fn nrm2_sq(x: &[f64]) -> f64 {
        x.iter().map(|a| a * a).sum()
    }

    #[inline]
    pub fn dot(s: &[f64], x: &[f64]) -> f64 {
        s.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
    }

    #[inline]
    pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
        for (i, &j) in y.iter_mut().zip(x.iter()) {
            *i = a.mul_add(j, *i)
        }
    }
}

#[allow(non_snake_case)]
fn solve_l2r_l1l2_svc(prob: &Problem, param: &Parameter) -> (usize, Vec<f64>) {
    let l = prob.l;
    let eps = param.eps;
    let mut i;
    let mut iter = 0;
    let mut s;
    let mut d;
    let mut G;
    let mut alpha_old;
    let mut intercept = 0.0;
    let mut w = vec![0.0; prob.n];

    let mut QD: Vec<f64> = vec![0.0; l];
    let mut index: Vec<usize> = (0..l).collect();
    let mut alpha: Vec<f64> = vec![0.0; l];
    let mut active_size = l;

    // PG: projected gradient, for shrinking and stopping
    let mut PG: f64;
    let mut PGmax_old = f64::INFINITY;
    let mut PGmin_old = f64::NEG_INFINITY;
    let mut PGmax_new;
    let mut PGmin_new;

    for i in 0..l {
        QD[i] = BLASOperator::nrm2_sq(prob.x[i]) + prob.bias * prob.bias;
        BLASOperator::axpy(prob.y[i] as f64 * alpha[i], prob.x[i], &mut w);
        intercept += prob.y[i] as f64 * alpha[i] * prob.bias;
    }

    // Create the default RNG.
    let mut rng = MT19937::from_seed(param.random_seed);
    while iter < param.max_iter {
        PGmax_new = f64::NEG_INFINITY;
        PGmin_new = f64::INFINITY;

        for i in 0..active_size {
            let j = i + rng.bounded_rand_int((active_size - i) as u32) as usize;
            index.swap(i, j)
        }
        s = 0;
        while s < active_size {
            i = index[s];

            G = prob.y[i]
                * (BLASOperator::dot(&w, prob.x[i]) + intercept * prob.bias)
                - 1.0;
            PG = 0.0;
            if alpha[i] == 0.0 {
                if G > PGmax_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if G < 0.0 {
                    PG = G
                }
            } else if alpha[i] == param.C {
                if G < PGmin_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if G > 0.0 {
                    PG = G
                }
            } else {
                PG = G
            }
            PGmax_new = PGmax_new.max(PG);
            PGmin_new = PGmin_new.min(PG);

            if PG.abs() > 1.0e-12 {
                alpha_old = alpha[i];
                alpha[i] = (alpha[i] - G / QD[i]).clamp(0.0, param.C);
                d = (alpha[i] - alpha_old) * prob.y[i];
                BLASOperator::axpy(d, prob.x[i], &mut w);
                intercept += d * prob.bias;
            }

            s += 1;
        }
        iter += 1;

        if PGmax_new - PGmin_new <= eps
            && PGmax_new.abs() <= eps
            && PGmin_new.abs() <= eps
        {
            if active_size == l {
                break;
            } else {
                active_size = l;
                PGmax_old = f64::INFINITY;
                PGmin_old = f64::NEG_INFINITY;
                continue;
            }
        }
        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if PGmax_old <= 0.0 {
            PGmax_old = f64::INFINITY
        }
        if PGmin_old >= 0.0 {
            PGmin_old = f64::NEG_INFINITY
        }
    }
    if prob.bias != 0.0 {
        w.push(intercept);
    }
    (iter, w)
}

pub fn train(prob: &Problem, param: Parameter) -> (usize, Vec<f64>) {
    // reorganise training data into consecutive labels
    // group training data of the same class
    let no_neg = prob.y.iter().filter(|&&a| a <= 0.0).count();
    let mut neg_index = Vec::with_capacity(no_neg);
    let mut pos_index = Vec::with_capacity(prob.l - no_neg);
    for (ind, &val) in prob.y.iter().enumerate() {
        if val <= 0.0 {
            neg_index.push(ind)
        } else {
            pos_index.push(ind)
        }
    }
    neg_index.append(&mut pos_index);

    // construct subproblem
    let x = neg_index.iter().map(|&ind| prob.x[ind]).collect();

    let mut y = vec![-1.0; no_neg];
    y.append(&mut vec![1.0; prob.l - no_neg]);

    let sub_prob = Problem {
        l: prob.l,
        n: prob.n,
        y: &y[..],
        x,
        bias: prob.bias,
    };

    solve_l2r_l1l2_svc(&sub_prob, &param)
}
