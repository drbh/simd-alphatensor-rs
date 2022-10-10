use criterion::{black_box, criterion_group, criterion_main, Criterion};

use simd_alphatensor_rs::{
    multiply_2_by_2_matrix_a_with_2_by_2_matrix_b, multiply_4_by_4_matrix_a_with_4_by_4_matrix_b,
};

use ndarray::arr2;

fn traditional_2x2x2(a: [i32; 4], b: [i32; 4]) {
    let a = arr2(&[[a[0], a[1]], [a[2], a[3]]]);
    let b = arr2(&[[b[0], b[1]], [b[2], b[3]]]);
    a.dot(&b);
}

fn traditional_4x4x4(a: [i32; 16], b: [i32; 16]) {
    let a = arr2(&[
        [a[0], a[1], a[2], a[3]],
        [a[4], a[5], a[6], a[7]],
        [a[8], a[9], a[10], a[11]],
        [a[12], a[13], a[14], a[15]],
    ]);
    let b = arr2(&[
        [b[0], b[1], b[2], b[3]],
        [b[4], b[5], b[6], b[7]],
        [b[8], b[9], b[10], b[11]],
        [b[12], b[13], b[14], b[15]],
    ]);
    a.dot(&b);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("simd_alphatensor_2x2x2", |b| {
        b.iter(|| {
            multiply_2_by_2_matrix_a_with_2_by_2_matrix_b(
                black_box([1000, 2000, 3000, 4000]),
                black_box([3000, 4000, 5000, 6000]),
            )
        })
    });
    c.bench_function("ndarry_2x2x2", |b| {
        b.iter(|| {
            traditional_2x2x2(
                black_box([1000, 2000, 3000, 4000]),
                black_box([3000, 4000, 5000, 6000]),
            )
        })
    });
    c.bench_function("simd_alphatensor_4x4x4", |b| {
        b.iter(|| {
            multiply_4_by_4_matrix_a_with_4_by_4_matrix_b(
                black_box([
                    1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000,
                    2000, 3000, 4000,
                ]),
                black_box([
                    1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000,
                    2000, 3000, 4000,
                ]),
            )
        })
    });
    c.bench_function("ndarry_4x4x4", |b| {
        b.iter(|| {
            traditional_4x4x4(
                black_box([
                    1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000,
                    2000, 3000, 4000,
                ]),
                black_box([
                    1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000, 2000, 3000, 4000, 1000,
                    2000, 3000, 4000,
                ]),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
