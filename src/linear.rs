use crate::MT19937::MT19937;
use std::convert::TryFrom;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Problem<'a> {
    l: usize,          // number of training data
    n: usize,          // number of features (including the bias feature if bias >= 0)
    y: &'a [f64], // array of target values (OPT bool for classification and f64 for regression)
    x: Vec<&'a [f64]>, // array of sparsely represented traning vectors
    bias: f64,    // < 0 if no bias term
}

impl<'a> Problem<'a> {
    pub fn new(l: usize, n: usize, y: &'a [f64], x: Vec<&'a [f64]>, bias: f64) -> Self {
        Self { l, n, y, x, bias }
    }
}

#[derive(PartialEq, Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum SolverType {
    L2R_LR,              // L2-regularized logistic regression (primal)
    L2R_L2LOSS_SVC_DUAL, // L2-regularized L2-loss support vector classification (dual)
    L2R_L2LOSS_SVC,      // L2-regularized L2-loss support vector classification (primal)
    L2R_L1LOSS_SVC_DUAL, // L2-regularized L1-loss support vector classification (dual)
    MCSVM_CS,            // support vector classification by Crammer and Singer
    L1R_L2LOSS_SVC,      // L1-regularized L2-loss support vector classification
    L1R_LR,              // L1-regularized logistic regression
    L2R_LR_DUAL,         // L2-regularized logistic regression (dual)
    L2R_L2LOSS_SVR = 11, // L2-regularized L2-loss support vector regression (primal)
    L2R_L2LOSS_SVR_DUAL, // L2-regularized L2-loss support vector regression (dual)
    L2R_L1LOSS_SVR_DUAL, // L2-regularized L1-loss support vector regression (dual)
    ONECLASS_SVM = 21,   // one-class support vector machine (dual)
}

impl TryFrom<usize> for SolverType {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::L2R_LR),
            1 => Ok(Self::L2R_L2LOSS_SVC_DUAL),
            2 => Ok(Self::L2R_L2LOSS_SVC),
            3 => Ok(Self::L2R_L1LOSS_SVC_DUAL),
            4 => Ok(Self::MCSVM_CS),
            5 => Ok(Self::L1R_L2LOSS_SVC),
            6 => Ok(Self::L1R_LR),
            7 => Ok(Self::L2R_LR_DUAL),
            11 => Ok(Self::L2R_L2LOSS_SVR),
            12 => Ok(Self::L2R_L2LOSS_SVR_DUAL),
            13 => Ok(Self::L2R_L1LOSS_SVR_DUAL),
            21 => Ok(Self::ONECLASS_SVM),
            _ => Err("Solver type does not exist"),
        }
    }
}

/// nr_weight, weight_label, and weight are used to change the penalty
/// for some classes (If the weight for a class is not changed, it is
/// set to 1). This is useful for training classifier using unbalanced
/// input data or with asymmetric misclassification cost.
pub struct Parameter {
    pub solver_type: SolverType,

    // these are for training only
    pub eps: f64, // stopping tolerance
    pub C: f64,   // cost of constraints violation
    // pub nr_weight: i32,    // number of elements in the array weight_label and weight
    // pub weight_label: i32, // Each weight[i] corresponds to weight_label[i], meaning that
    // // the penalty of class weight_label[i] is scaled by a factor of weight[i].
    // pub weight: f64,
    // pub p: f64,        // sensitiveness of loss of support vector regression
    // pub nu: f64,       // approximates the fraction of data as outliers
    // pub init_sol: f64, // the initial weight vectors
    // pub regularize_bias: i32,
    pub max_iter: usize,
    pub random_seed: u32,
}

pub struct Model {
    pub param: Parameter,
    pub nr_class: usize,   // number of classes; nr_class = 2 for regression.
    pub nr_feature: usize, // number of features
    pub w: Vec<f64>, // feature weights; its size is nr_feature*nr_class but is nr_feature if nr_class = 2
    pub label: [i8; 2], // label of each class
    pub bias: f64,
    pub rho: f64, // one-class SVM only
    pub intercept: f64,
}

impl Model {
    fn new(param: Parameter) -> Self {
        Self {
            param,
            nr_class: 0,
            nr_feature: 0,
            w: Vec::new(),
            label: [0, 0],
            bias: 1.0,
            rho: 0.0,
            intercept: 0.0,
        }
    }
}

pub struct SparseOperator;

impl SparseOperator {
    #[inline]
    pub fn nrm2_sq(x: &[f64]) -> f64 {
        x.iter().map(|a| a * a).sum()
    }

    #[inline]
    pub fn dot(s: &Vec<f64>, x: &[f64]) -> f64 {
        s.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
    }

    #[inline]
    pub fn axpy(a: f64, x: &[f64], y: &mut Vec<f64>) {
        for (i, j) in y.iter_mut().zip(x.iter()) {
            *i += a * j
        }
    }
}

fn solve_l2r_l1l2_svc(prob: &Problem, param: &Parameter, w: &mut Vec<f64>) -> usize {
    let l = prob.l;
    // let w_size = prob.n;
    let eps = param.eps;
    // let solver_type = param.solver_type;
    let mut i;
    let mut iter = 0;
    let mut s;
    let mut C;
    let mut d;
    let mut G;
    let mut alpha_old;
    let mut intercept = 0.0;

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

    // default solver_type: L2R_L2LOSS_SVC_DUAL
    // let diag = [0.0, 0.0, 0.0];
    let upper_bound = [param.C, 0.0, param.C];

    for i in 0..l {
        let xi = prob.x[i];
        QD[i] = SparseOperator::nrm2_sq(xi) + prob.bias * prob.bias;
        SparseOperator::axpy(prob.y[i] as f64 * alpha[i], xi, w);
        intercept += prob.y[i] as f64 * alpha[i] * prob.bias;

        // moved index sequential assigning to above
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
            let yi = prob.y[i];
            let xi = &prob.x[i];

            G = yi as f64 * (SparseOperator::dot(w, xi) + intercept * prob.bias) - 1f64;
            C = upper_bound[(prob.y[i] as i8 + 1) as usize];
            PG = 0.0;
            if alpha[i] == 0.0 {
                if G > PGmax_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if G < 0.0 {
                    PG = G
                }
            } else if alpha[i] == C {
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
                alpha[i] = (alpha[i] - G / QD[i]).max(0.0).min(C);
                d = (alpha[i] - alpha_old) * yi as f64;
                SparseOperator::axpy(d, xi, w);
                intercept += d * prob.bias;
            }

            s += 1;
        }
        iter += 1;
        // if iter % 10 == 0 {
        //     print!(".")
        // }

        if PGmax_new - PGmin_new <= eps && PGmax_new.abs() <= eps && PGmin_new.abs() <= eps {
            if active_size == l {
                break;
            } else {
                active_size = l;
                print!("*");
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
    // print!("\noptimization finished, #iter = {}\n", iter);

    // let v =
    //     w.iter().map(|a| a * a).sum::<f64>() + alpha.iter().sum::<f64>() * -2.0;
    // let nSV = alpha.iter().filter(|&&a| a > 0.0).count();

    // println!("Objective value = {}", v / 2.0);
    // println!("nSV = {}", nSV);
    w.push(intercept);
    iter
}

pub fn train(prob: &Problem, param: Parameter) -> (Model, usize) {
    let mut model = Model::new(param);
    let l = prob.l;
    let n = prob.n;
    let w_size = prob.n;
    if prob.bias >= 0.0 {
        model.nr_feature = n - 1
    } else {
        model.nr_feature = n
    }
    model.bias = prob.bias;

    // group training data of the same class
    let (nr_class, label, start, count, perm) = group_classes(prob);

    model.nr_class = nr_class;
    model.label = label;

    // construct subproblem
    let x = perm.iter().map(|&ind| prob.x[ind]).collect();

    model.w = vec![0.0; w_size];
    let mut y = vec![-1.0; start[1]];
    y.append(&mut vec![1.0; count[1]]);
    let sub_prob = Problem {
        l,
        n,
        y: &y[..],
        x,
        bias: prob.bias,
    };

    let n_iter;
    if model.param.solver_type == SolverType::L2R_L1LOSS_SVC_DUAL {
        n_iter = solve_l2r_l1l2_svc(&sub_prob, &model.param, &mut model.w);
    } else {
        panic!("Unsupported solver type")
    }
    (model, n_iter)
}

// group_classes reorganise training data into consecutive labels
fn group_classes(prob: &Problem) -> (usize, [i8; 2], [usize; 2], [usize; 2], Vec<usize>) {
    let no_neg = prob.y.iter().filter(|&&a| a <= 0.0).count();
    let start = [0, no_neg];
    let count = [no_neg, prob.l - no_neg];
    let mut neg = Vec::with_capacity(no_neg);
    let mut pos = Vec::with_capacity(prob.l - no_neg);
    // TODO bench against filter map
    for (ind, &val) in prob.y.iter().enumerate() {
        if val <= 0.0 {
            neg.push(ind)
        } else {
            pos.push(ind)
        }
    }
    neg.append(&mut pos);
    (2, [1, -1], start, count, neg)
}
