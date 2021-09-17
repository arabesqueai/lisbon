// unsafe as the sum can under/overflow
fn dnrm2_unsafe(vector: &Vec<f64>) -> f64 {
    vector.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

// Implementing the Blue's algorithm
fn dnrm2(vector: &Vec<f64>) -> f64 {
    unimplemented!()
}
