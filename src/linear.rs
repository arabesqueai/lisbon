use rand::{thread_rng, Rng};
use std::fmt::Debug;
use std::process::exit;
use std::{fs, io, io::BufRead};

#[derive(Clone)]
struct FeatureNode {
    index: usize,
    value: f64,
}

impl Debug for FeatureNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}, {})", self.index, self.value))
    }
}

impl FeatureNode {
    fn new(input: &str) -> Self {
        if let Some(ind_val) = input.split_once(":") {
            return Self {
                index: ind_val.0.parse::<usize>().unwrap(),
                value: ind_val.1.parse::<f64>().unwrap(),
            };
        } else {
            panic!(format!("Malformed input {}", input))
        }
    }
}

#[derive(Debug)]
struct Problem {
    l: usize,                 // number of training data
    n: usize, // number of features (including the bias feature if bias >= 0)
    y: Vec<i8>, // array of target values (OPT bool for classification and f64 for regression)
    x: Vec<Vec<FeatureNode>>, // array of sparsely represented traning vectors
    bias: f64,  // < 0 if no bias term
}

#[derive(PartialEq, Clone, Copy)]
#[allow(non_camel_case_types)]
enum SolverType {
    L2R_LR,              // L2-regularized logistic regression (primal)
    L2R_L2LOSS_SVC_DUAL, // L2-regularized L2-loss support vector classification (dual)
    L2R_L2LOSS_SVC, // L2-regularized L2-loss support vector classification (primal)
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

/// nr_weight, weight_label, and weight are used to change the penalty
/// for some classes (If the weight for a class is not changed, it is
/// set to 1). This is useful for training classifier using unbalanced
/// input data or with asymmetric misclassification cost.
struct Parameter {
    solver_type: SolverType,

    // these are for training only
    eps: f64,          // stopping tolerance
    C: f64,            // cost of constraints violation
    nr_weight: i32, // number of elements in the array weight_label and weight
    weight_label: i32, // Each weight[i] corresponds to weight_label[i], meaning that
    // the penalty of class weight_label[i] is scaled by a factor of weight[i].
    weight: f64,
    p: f64,        // sensitiveness of loss of support vector regression
    nu: f64,       // approximates the fraction of data as outliers
    init_sol: f64, // the initial weight vectors
    regularize_bias: i32,
}

struct Model {
    param: Parameter,
    nr_class: usize, // number of classes; nr_class = 2 for regression.
    nr_feature: usize, // number of features
    w: Vec<f64>, // feature weights; its size is nr_feature*nr_class but is nr_feature if nr_class = 2
    label: [i8; 2], // label of each class
    bias: f64,
    rho: f64, // one-class SVM only
}

impl Model {
    fn new(param: Parameter) -> Self {
        Self {
            param,
            nr_class: 0,
            nr_feature: 0,
            w: Vec::new(),
            label: [0, 0],
            bias: 0.0,
            rho: 0.0,
        }
    }
}

struct SparseOperator;

impl SparseOperator {
    fn nrm2_sq(x: &Vec<FeatureNode>) -> f64 {
        // skip last element which would have index -1
        // removed for now
        x[..x.len()].iter().map(|a| a.value * a.value).sum()
    }

    fn dot(s: &Vec<f64>, x: &Vec<FeatureNode>) -> f64 {
        x[..x.len()].iter().map(|a| s[a.index - 1] * a.value).sum()
    }

    fn sparse_dot(x1: &Vec<FeatureNode>, x2: &Vec<FeatureNode>) -> f64 {
        let mut iter_x1 = x1[..x1.len()].iter();
        let mut iter_x2 = x2[..x2.len()].iter();
        let mut val1 = iter_x1.next();
        let mut val2 = iter_x2.next();
        let mut a1;
        let mut a2;
        let mut ret = 0.0;
        while val1.is_some() && val2.is_some() {
            a1 = val1.unwrap();
            a2 = val2.unwrap();
            if a1.index == a2.index {
                ret += val1.unwrap().value * val2.unwrap().value;
                val1 = iter_x1.next();
                val2 = iter_x2.next();
            } else if a1.index > a2.index {
                val2 = iter_x2.next()
            } else {
                val1 = iter_x1.next()
            }
        }
        ret
    }

    fn axpy(a: f64, x: &Vec<FeatureNode>, y: &mut Vec<f64>) {
        for i in x[..x.len()].iter() {
            y[i.index - 1] += a * i.value
        }
    }
}

fn solve_l2r_l1l2_svc(
    prob: &Problem,
    param: &Parameter,
    w: &mut Vec<f64>,
    Cp: f64,
    Cn: f64,
    max_iter: usize,
) -> usize {
    let l = prob.l;
    let w_size = prob.n;
    let eps = param.eps;
    let solver_type = param.solver_type;
    let (mut i, mut iter) = (0, 0);
    let mut s;
    let mut C;
    let mut d;
    let mut G;

    let mut QD: Vec<f64> = vec![0.0; l];
    let mut index: Vec<usize> = (0..l).collect();
    let mut alpha: Vec<f64> = vec![0.0; l];
    // let mut y: Vec<i8> = Vec::with_capacity(l);
    // TODO: benchmark preallocation
    let mut active_size = l;

    // PG: projected gradient, for shrinking and stopping
    let mut PG: f64;
    let mut PGmax_old = f64::INFINITY;
    let mut PGmin_old = f64::NEG_INFINITY;
    let mut PGmax_new;
    let mut PGmin_new;

    // default solver_type: L2R_L2LOSS_SVC_DUAL
    let mut diag = [0.5 / Cn, 0.0, 0.5 / Cp];
    let mut upper_bound = [f64::INFINITY, 0.0, f64::INFINITY];
    if solver_type == SolverType::L2R_L1LOSS_SVC_DUAL {
        diag = [0.0, 0.0, 0.0];
        upper_bound = [Cn, 0.0, Cp];
    }

    // let y = prob.y;

    // Initial alpha can be set here. Note that
    // 0 <= alpha[i] <= upper_bound[GETI(i)]
    for i in 0..w_size {
        w[i] = 0.0
    }

    for i in 0..l {
        let xi = &prob.x[i];
        QD[i] = diag[(prob.y[i] + 1) as usize] + SparseOperator::nrm2_sq(xi);
        SparseOperator::axpy(prob.y[i] as f64 * alpha[i], xi, w);
        // moved index sequential assigning to above
    }

    let mut rng = thread_rng();
    while iter < max_iter {
        PGmax_new = f64::NEG_INFINITY;
        PGmin_new = f64::INFINITY;

        for i in 0..active_size {
            // let j = i + rng.gen::<usize>() % (active_size - i);
            let j = i + 100 % (active_size - i);
            index.swap(i, j)
        }
        s = 0;
        while s < active_size {
            i = index[s];
            let yi = prob.y[i];
            let xi = &prob.x[i];

            G = yi as f64 * SparseOperator::dot(w, xi) - 1f64
                + alpha[i] * diag[(prob.y[i] + 1) as usize];
            C = upper_bound[(prob.y[i] + 1) as usize];
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
                let alpha_old = alpha[i];
                alpha[i] = (alpha[i] - G / QD[i]).max(0.0).min(C);
                d = (alpha[i] - alpha_old) * yi as f64;
                SparseOperator::axpy(d as f64, xi, w);
            }

            s += 1;
        }
        iter += 1;
        if iter % 10 == 0 {
            print!(".")
        }

        if PGmax_new - PGmin_new <= eps
            && PGmax_new.abs() <= eps
            && PGmin_new.abs() <= eps
        {
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
    print!("\noptimization finished, #iter = {}\n", iter);

    let mut v: f64 = w.iter().map(|a| a * a).sum();
    let mut nSV = 0; // number of support vectors
    for i in 0..l {
        v += alpha[i] * (alpha[i] * diag[(prob.y[i] + 1) as usize] - 2.0);
        if alpha[i] > 0.0 {
            nSV += 1;
        }
    }

    println!("Objective value = {}", v / 2.0);
    println!("nSV = {}", nSV);

    iter
}

fn train(prob: &Problem, param: Parameter) -> Model {
    let mut model = Model::new(param);
    let l = prob.l;
    let n = prob.n;
    let w_size = prob.n;
    let i: i32;
    let j: i32;
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

    // calculate weighted C (skipped, MVP do not deal with class weighing)
    let weighted_C = [model.param.C, model.param.C];

    // construct subproblem
    let x: Vec<Vec<FeatureNode>> =
        perm.iter().map(|&ind| prob.x[ind].clone()).collect();

    let mut sub_prob = Problem {
        l,
        n,
        y: vec![0; l],
        x,
        bias: prob.bias,
    };

    model.w = vec![0.0; w_size];
    let mut sub_y = vec![1; start[1]];
    sub_y.append(&mut vec![-1; count[1]]);
    sub_prob.y = sub_y;

    train_one(
        &sub_prob,
        &model.param,
        &mut model.w,
        weighted_C[0],
        weighted_C[1],
    );

    model
}

fn train_one(
    prob: &Problem,
    param: &Parameter,
    w: &mut Vec<f64>,
    Cp: f64,
    Cn: f64,
) {
    // we have the same penalty for both direction
    let mut C = vec![Cp; prob.l];
    let mut pos = prob.y.iter().filter(|&&a| a > 0).count();
    let mut neg = prob.l - pos;
    if param.solver_type == SolverType::L2R_L1LOSS_SVC_DUAL {
        let iter = solve_l2r_l1l2_svc(prob, param, w, Cp, Cn, 300);
        if iter >= 300 {
            print!("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n")
        };
    }
}

// group_classes reorganise training data into consecutive labels
fn group_classes(
    prob: &Problem,
) -> (usize, [i8; 2], [usize; 2], [usize; 2], Vec<usize>) {
    let no_pos = prob.y.iter().filter(|&&a| a > 0).count();
    let start = [0, no_pos];
    let count = [no_pos, prob.l - no_pos];
    let mut pos = Vec::with_capacity(no_pos);
    let mut neg = Vec::with_capacity(prob.l - no_pos);
    // TODO bench against filter map
    for (ind, &val) in prob.y.iter().enumerate() {
        if val > 0 {
            pos.push(ind)
        } else {
            neg.push(ind)
        }
    }
    pos.append(&mut neg);
    (2, [1, -1], start, count, pos)
}

fn read_file(filename: &str) -> Problem {
    let file = fs::File::open(filename).unwrap();
    let lines = io::BufReader::new(file).lines();
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut n = 0usize;
    for line in lines {
        if let Ok(l) = line {
            let mut row = Vec::new();
            let mut elements = l.split_whitespace();
            if let Some(elem) = elements.next() {
                y.push(elem.parse::<i8>().unwrap())
            }
            for elem in elements {
                row.push(FeatureNode::new(elem))
            }
            let last_index = row.last().unwrap().index;
            if last_index > n {
                n = last_index
            }
            // row.push(FeatureNode { index: -1, value: 0.0 });
            x.push(row)
        }
    }
    Problem {
        l: x.len(),
        n,
        y,
        x,
        bias: -1.0,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn group() {
        let prob = read_file("heart_scale");
        train(
            &prob,
            Parameter {
                solver_type: SolverType::L2R_L1LOSS_SVC_DUAL,
                eps: 0.1,
                C: 1.0,
                nr_weight: 0,
                weight_label: 0,
                weight: 0.0,
                p: 0.0,
                nu: 0.0,
                init_sol: 0.0,
                regularize_bias: 0,
            },
        );
    }
}
