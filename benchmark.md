# Benchmark results

As with all benchmarks, your mileage may vary :)

## With PMLB's adult dataset

48842 rows of data and 14 features. See `bench.py`.

| library | time (s) | memory (MB) |
| :-: | :-: | :-: |
| liblinear | 6.473 | 137.4 |
| lisbon with `cpu=native` | 3.725 | 127.9 |
| lisbon without `cpu=native` | 5.784 | 127.8 |

<details><summary>Click for details</summary>
<p>

`liblinear`

```
liblinear took 6.473016023635864 seconds and 1000 iterations
last 10 coefficients:  [[ 0.00066071  0.00200457  0.0010956   0.00202347  0.00089573 -0.00021549
  -0.00014445 -0.00031917  0.00087296  0.01139567]]
Intercept:  [0.00033345]
Max memory usage:  137.41015625
```

With `RUSTFLAGS='-C target-cpu=native'`

```
lisbon took 3.725350856781006 seconds and 1000 iterations
last 10 coefficients:  [[ 0.00066071  0.00200457  0.0010956   0.00202347  0.00089573 -0.00021549
  -0.00014445 -0.00031917  0.00087296  0.01139567]]
Intercept:  [0.00033345]
Max memory usage:  127.85546875
```

Without `RUSTFLAGS='-C target-cpu=native'`

```
lisbon took 5.783940315246582 seconds and 1000 iterations
last 10 coefficients:  [[ 0.00066071  0.00200457  0.0010956   0.00202347  0.00089573 -0.00021549
  -0.00014445 -0.00031917  0.00087296  0.01139567]]
Intercept:  [0.00033345]
Max memory usage:  127.75390625
```
  
</p>
</details>

## With Arabesque's dataset

210447 rows of data and 1000 features.

| library | time (s) | memory (MB) |
| :-: | :-: | :-: |
| liblinear | 41.096 | 5765.0 |
| lisbon with `cpu=native` | 25.569 | 2539.4 |
| lisbon without `cpu=native` | 38.162 | 2538.5 |

<details><summary>Click for details</summary>
<p>

`liblinear`

```
liblinear took 41.09577989578247 seconds and 8200 iterations
last 10 coefficients:  [[ 2.16037238  0.47328366  2.19392385  0.87080528  1.58050211 -0.61958832
  -0.70398499  0.1571059  -0.79993815 -3.43818232]]
Intercept:  [-0.14189001]
Max memory usage:  5765.00390625
```

With `RUSTFLAGS='-C target-cpu=native'`

```
lisbon took 25.569344520568848 seconds and 8200 iterations
last 10 coefficients:  [[ 2.16037238  0.47328366  2.19392385  0.87080528  1.58050211 -0.61958832
  -0.70398499  0.1571059  -0.79993815 -3.43818232]]
Intercept:  [-0.14189001]
Max memory usage:  2539.4375

```

Without `RUSTFLAGS='-C target-cpu=native'`

```
lisbon took 38.161619424819946 seconds and 8200 iterations
last 10 coefficients:  [[ 2.16037238  0.47328366  2.19392385  0.87080528  1.58050211 -0.61958832
  -0.70398499  0.1571059  -0.79993815 -3.43818232]]
Intercept:  [-0.14189001]
Max memory usage:  2538.50390625
```
  
    
</p>
</details>

## With Arabesque's bigger dataset

747922 rows of data and 1000 features.


| library | time (s) | memory (MB) |
| :-: | :-: | :-: |
| liblinear | 84.259 | 20205.0 |
| lisbon with `cpu=native` | 50.791 | 8742.1 |
| lisbon without `cpu=native` | 76.959 | 8741.7 |

<details><summary>Click for details</summary>
<p>

`liblinear`

```
liblinear took 84.25864148139954 seconds and 10000 iterations
last 10 coefficients:  [[-2.03525229  2.05070747  0.26083448  0.97163776  2.89743793  1.46319187
   3.3082313  -3.31360901  0.31717845 -4.85035968]]
Intercept:  [0.55134394]
Max memory usage:  20204.97265625
```

With `RUSTFLAGS='-C target-cpu=native'`

```
lisbon took 50.790966272354126 seconds and 10000 iterations
last 10 coefficients:  [[-2.03525229  2.05070747  0.26083448  0.97163776  2.89743793  1.46319187
   3.3082313  -3.31360901  0.31717845 -4.85035968]]
Intercept:  [0.55134394]
Max memory usage:  8742.1484375
```

Without `RUSTFLAGS='-C target-cpu=native'`

```
lisbon took 76.95874190330505 seconds and 10000 iterations
last 10 coefficients:  [[-2.03525229  2.05070747  0.26083448  0.97163776  2.89743793  1.46319187
   3.3082313  -3.31360901  0.31717845 -4.85035968]]
Intercept:  [0.55134394]
Max memory usage:  8741.66015625
```
