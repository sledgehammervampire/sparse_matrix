use std::{ffi::OsStr, fs::read_to_string};

use criterion::{criterion_group, criterion_main, Criterion};
use num::traits::NumAssign;
use spam::{
    csr_matrix::CsrMatrix,
    dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
    Matrix,
};
use walkdir::WalkDir;

pub fn bench_mul_heap(c: &mut Criterion) {
    fn foo<T: Clone + NumAssign + Send + Sync>(c: &mut Criterion, f: &OsStr, m1: DokMatrix<T>) {
        let m1 = CsrMatrix::from(m1);
        if let Some(&len) = m1.row_nnz_freq().keys().last() {
            if len > 10000 {
                return;
            }
        }
        let m2 = m1.clone();
        c.bench_function(
            &format!(
                "bench mul_heap {:?} ({}x{}, {} nonzero entries)",
                f,
                m1.rows(),
                m1.cols(),
                m1.nnz()
            ),
            |b| b.iter(|| m1.mul_heap(&m2)),
        );
    }

    for entry in WalkDir::new("matrices") {
        let entry = entry.unwrap();
        if let Some(ext) = entry.path().extension() {
            if ext == "mtx" {
                let f = entry.path().file_name().unwrap();
                match parse_matrix_market::<i64, f64>(&read_to_string(entry.path()).unwrap())
                    .unwrap()
                    .1
                {
                    MatrixType::Integer(m1) => {
                        foo(c, f, m1);
                    }
                    MatrixType::Real(m1) => {
                        foo(c, f, m1);
                    }
                    MatrixType::Complex(m1) => {
                        foo(c, f, m1);
                    }
                }
            }
        }
    }
}

criterion_group!(benches, bench_mul_heap);
criterion_main!(benches);
