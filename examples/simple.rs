use simd_alphatensor_rs::{
    multiply_2_by_2_matrix_a_with_2_by_2_matrix_b, multiply_4_by_4_matrix_a_with_4_by_4_matrix_b,
};

fn main() {
    let result = multiply_2_by_2_matrix_a_with_2_by_2_matrix_b(
        [1000, 2000, 3000, 4000],
        [3000, 4000, 5000, 6000],
    );
    println!("Example running {:?}", result);

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
    println!("Example running {:?}", result)
}
