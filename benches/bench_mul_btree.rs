use std::{ffi::OsStr, fs::read_to_string};

use criterion::{criterion_group, criterion_main, Criterion};
use num::traits::NumAssign;
use spam::{
    csr_matrix::CsrMatrix,
    dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
    Matrix,
};
use walkdir::WalkDir;

pub fn bench_mul_btree(c: &mut Criterion) {
    fn foo<T: Clone + NumAssign + Send + Sync>(c: &mut Criterion, f: &OsStr, m: DokMatrix<T>) {
        let m = CsrMatrix::from(m);
        c.bench_function(
            &format!(
                "bench mul_btree {:?} ({}x{}, {} nonzero entries)",
                f,
                m.rows(),
                m.cols(),
                m.nnz()
            ),
            |b| b.iter(|| m.mul_btree(&m)),
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
                    MatrixType::Integer(m) => {
                        foo(c, f, m);
                    }
                    MatrixType::Real(m) => {
                        foo(c, f, m);
                    }
                    MatrixType::Complex(m) => {
                        foo(c, f, m);
                    }
                }
            }
        }
    }
}

criterion_group!(benches, bench_mul_btree);
criterion_main!(benches);
