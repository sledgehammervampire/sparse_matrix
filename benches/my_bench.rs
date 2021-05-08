use std::fs::{read_dir, read_to_string};

use criterion::{criterion_group, criterion_main, Criterion};
use spam::{
    csr_matrix::CsrMatrix,
    dok_matrix::{parse_matrix_market, MatrixType},
    Matrix,
};

pub fn bench_mul(c: &mut Criterion) {
    for f in read_dir("matrices").unwrap() {
        let f = f.unwrap();
        match parse_matrix_market::<i32, f32>(&read_to_string(f.path()).unwrap())
            .unwrap()
            .1
        {
            MatrixType::Integer(m1) => {
                let m1 = CsrMatrix::from(m1);
                let m2 = m1.clone();
                c.bench_function(
                    &format!(
                        "bench csr mul {:?} ({}x{}, {} entries)",
                        f,
                        m1.rows(),
                        m1.cols(),
                        m1.nnz()
                    ),
                    |b| b.iter(|| &m1 * &m2),
                );
            }
            MatrixType::Real(m1) => {
                let m1 = CsrMatrix::from(m1);
                let m2 = m1.clone();
                c.bench_function(
                    &format!(
                        "bench csr mul {:?} ({}x{}, {} entries)",
                        f,
                        m1.rows(),
                        m1.cols(),
                        m1.nnz()
                    ),
                    |b| b.iter(|| &m1 * &m2),
                );
            }
        };
    }
}

criterion_group!(benches, bench_mul);
criterion_main!(benches);
