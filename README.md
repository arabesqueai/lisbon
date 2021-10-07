# lisbon

_DO NOT USE_ if your function arguments do not look like `svm.LinearSVC(loss="hinge")`.

`lisbon` aims to be a drop-in replacement for `liblinear` which `scikit-learn` leaverages for linear classification problems, currently only supports L2 regularised hinge loss by solving the dual problem (routine 3). The same APIs are provided as `scikit-learn`'s `liblinear` wrapper so you can monkey-patch `scikit-learn`'s svm library to use `lisbon`

```python
from sklearn import svm
import lisbon

svm._base.liblinear = lisbon
```

and the following computations will use `lisbon`. To switch back: `svm._base.liblinear = svm._liblinear`.

## Installation

- Make sure you have the Rust toolchain `rustc`, `cargo`, `rust-std` installed. The quickest way to do it is `curl https://sh.rustup.rs -sSf | sh -s`
  - For a minimal installation: `curl https://sh.rustup.rs -sSf | sh -s -- --profile minimal`
- With your desired Python environment, `pip install maturin`
- From `lisbon`'s project root, run `RUSTFLAGS='-C target-cpu=native' maturin develop --release` will install `lisbon` as a package to your Python environment
  - Note that the `RUSTFLAGS='-C target-cpu=native'` environmental variable ensures that Rustc compiles against your CPU's supported instruction sets to enable more SIMD optimisations (e.g. AVX2, FMA).
- For dev/benchmark purposes, consider installing the packages listed in `requirements-dev.txt`

## Limitations

Currently, `lisbon` only supports L2 regularised hinge loss and does not support

1. sample weights
2. class weights
3. different penalty `C` for labels

## Deviations from the source implementation

1. As with `scikit-learn`'s modification, the order of labels are flipped to be consistent with the rest of the `scikit-learn` family
   - `liblinear` uses [+1, -1] ordering
   - `scikit-learn` uses [-1, +1] ordering
2. Uses a MT19937 + tweaked Lemire post-processor to generate a random number within range

## Why is lisbon faster

- `liblinear` uses sparse matrix representation for the dot/norm operations, so `scikit-learn` needs to convert the dense numpy matrix to sparse first then pass to liblinear. Our encoded data is not sparse so that’s inefficient and prevents some simd optimisations
- By reading the numpy C array directly underneath there’s no need to copy/duplicate data so saves memory
- Specialised. Some array reads and computations are optimised away as we know what the values are for the specific problems

## Ref

1. [2-norm](https://dl.acm.org/doi/pdf/10.1145/3061665)
2. [A Dual Coordinate Descent Method for Large-scale Linear SVM](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)

### License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in lisbon by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
