# simd-alphatensor-rs

> tldr; alphatensor matrix breakthrough algorithims + simd + rust.

This repo contains the cutting edge matrix multiplication algorithms that were found by [alphatensor](https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor), and as far as I know is the first novel machine imagined algorithm ever 🦾 🧠 

This repo/library by default includes the first 25 algorithms and the one we are most interested in `multiply_4_by_4_matrix_a_with_4_by_4_matrix_b`. Additionally this implementations aims to minimize the number of multiplication steps and attempts to aggregate as many multiplication steps into a single SIMD vector when possible.

### ELI5

A super smart computer figured out how to do an important math problem in less steps than we knew was possible. By doing this math in less steps we can save time and electricity every time these things are used. And they're used trillions, yes trillions of times a day. 

### Example use

```rust
use simd_alphatensor_rs::{
    multiply_2_by_2_matrix_a_with_2_by_2_matrix_b, multiply_4_by_4_matrix_a_with_4_by_4_matrix_b,
};

// all arrays are unrolled row wise as in and output
fn main() {
    let result = multiply_2_by_2_matrix_a_with_2_by_2_matrix_b(
        [1000, 2000, 3000, 4000],
        [3000, 4000, 5000, 6000],
    );
    println!("Example of 2x2 * 2x2 {:?}", result);

    let result = multiply_4_by_4_matrix_a_with_4_by_4_matrix_b(
        [
            1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000,
            3000, 4000,
        ],
        [
            1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000,
            3000, 4000,
        ],
    );
    println!("Example of 4x4 * 4x4 {:?}", result)
}
```

### Benchmarks

Run with
```bash
cargo bench
```

| Benchmark      | Average Runtime |
| ----------- | ----------- |
| ndarry_2x2x2      | 314.53 ns       |
| simd_alphatensor_2x2x2   | 13.821 ns        |
| ndarry_4x4x4   | 402.60 ns        |
| simd_alphatensor_4x4x4   | 92.732 ns        |

```
➜  simd-alphatensor-rs (main) ✗ cargo bench
    Finished bench [optimized] target(s) in 0.13s
     Running unittests src/lib.rs (target/release/deps/simd_alphatensor_rs-115473bd3321baa9)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

Benchmarking simd_alphatensor_2x2x2: Collecting 100 samples in estimated 5.0000 s (3                                                                                    simd_alphatensor_2x2x2  time:   [13.798 ns 13.821 ns 13.847 ns]
                        change: [-4.8455% -3.8407% -2.9273%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 100 measurements (7.00%)
  4 (4.00%) high mild
  3 (3.00%) high severe

Benchmarking ndarry_2x2x2: Collecting 100 samples in estimated 5.0003 s (16M iterati                                                                                    ndarry_2x2x2            time:   [311.79 ns 314.53 ns 317.32 ns]
                        change: [-10.524% -8.5993% -6.6801%] (p = 0.00 < 0.05)
                        Performance has improved.

Benchmarking simd_alphatensor_4x4x4: Collecting 100 samples in estimated 5.0002 s (5                                                                                    simd_alphatensor_4x4x4  time:   [92.532 ns 92.732 ns 92.946 ns]
                        change: [-2.1432% -1.3174% -0.6179%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) high mild
  3 (3.00%) high severe

Benchmarking ndarry_4x4x4: Collecting 100 samples in estimated 5.0006 s (12M iterati                                                                                    ndarry_4x4x4            time:   [400.73 ns 402.60 ns 404.58 ns]
                        change: [-1.2978% -0.4004% +0.5315%] (p = 0.41 > 0.05)
                        No change in performance detected.
Found 3 outliers among 100 measurements (3.00%)
  1 (1.00%) high mild
  2 (2.00%) high severe
```

### Generation

You can manually generate this code via some scripts. Currently the algorithms are parsed by first converting the released numpy matrix files into python code. 🙏 big shout out to @https://github.com/99991 for this gem: [here](https://github.com/deepmind/alphatensor/issues/3). Once this python code is generated we can parse it into rust code and finally we can use the code as a library.

```bash
# fetch files from deepmind's github
make get-data

# convert alphatensors output
make gen

# parse python file and write to lib
cargo run -p codegen > src/gen.rs

# format the generated code
cargo fmt

# make sure it works with small example
cargo run --example simple

# profit 🙌
```

### Known limitations

1. This code is all experimental and is 2 steps removed from the output file shared by deepmind. We've tried our best to preserve the accuracy and the output is validated between steps. However please use this at you're own risk! and please please - do not use this in any production system!
2. Not all of the algorithms are ported into rust. Since the output results in vert large expanded rust functions, for demonstration purposes this library only ships with the following algorithm. This subset was selected since the specific `4x4x4` algorithm is likely the most applicable and also one of the algorithms that is superior to any known methods.

### Implementation considerations

```rust
// explicit funcs for each algo and explicit input and output sized arrays
// since everything is statically sized we can keep everything on the stack
pub fn multiply_2_by_2_matrix_a_with_2_by_2_matrix_b(a: [i32; 4], b: [i32; 4]) -> [i32; 4] {
    let [a11, a12, a21, a22] = a;
    let [b11, b12, b21, b22] = b;

    // we inline all of our multiplication steps in as few fixed sized 
    // 512-bit SIMD vector with 16 elements of type i32.
    // we don't always need all of the elements but in this impl we skipped
    // dynamically resizing the vector.
    let lefts = [i32x16::from([
        (a21 - a22),
        (a11 + a21 - a22),
        (a11 - a12 + a21 - a22),
        a12,
        (a11 + a21),
        a11,
        a22,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])];
    let rights = [i32x16::from([
        b12,
        (b12 + b21 + b22),
        (b21 + b22),
        b21,
        (b11 + b12 + b21 + b22),
        b11,
        (b12 + b22),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])];
    
    // here we do all of the multiplications above in only as many steps as 
    // simd vectors. In this case we only need one instruction (multiplication)
    // to perform all 7 multiplications needed for this algorithm
    let hs = [lefts[0] * rights[0]];

    // do the final summation steps
    let c11 = (hs[0][3] + hs[0][5]);
    let c12 = (-hs[0][1] + hs[0][4] - hs[0][5] - hs[0][6]);
    let c21 = (-hs[0][0] + hs[0][1] - hs[0][2] - hs[0][3]);
    let c22 = (hs[0][0] + hs[0][6]);

    // return new array
    return [c11, c12, c21, c22];
}
```

### TODOS

1. parse numpy directly in rust
2. include all (novel) algorithms
3. add a lot of benchmarking
4. clean up codegen
5. explain SIMD better
6. adjust SIMD array to minimize trailing
7. maybe SIMD config?
8. recursively make into SIMD only steps?
9. tests tests tests