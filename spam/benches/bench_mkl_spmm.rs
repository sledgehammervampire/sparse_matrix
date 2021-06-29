use std::{convert::TryFrom, fs};

use criterion::{criterion_group, criterion_main, Criterion};
use spam::{
    csr_matrix::{
        ffi::{CMklSparseMatrix, MklCsrMatrix, RustMklSparseMatrix},
        CsrMatrix,
    },
    dok_matrix::{parse_matrix_market, MatrixType},
    Matrix,
};
use walkdir::WalkDir;

macro_rules! inner {
    ($m:expr, $f:expr, $c:expr) => {
        let m: CsrMatrix<_, false> = CsrMatrix::from($m);
        let rows = m.rows();
        let cols = m.cols();
        let nnz = m.nnz();
        let mut m = MklCsrMatrix::try_from(m).unwrap();
        let m = CMklSparseMatrix::from(RustMklSparseMatrix::try_from(&mut m).unwrap());
        $c.bench_function(
            &format!(
                "bench mkl_spmm {:?} ({}x{}, {} nonzero entries)",
                $f, rows, cols, nnz
            ),
            |b| b.iter(|| &m * &m),
        );
    };
}

pub fn bench_mkl_spmm(c: &mut Criterion) {
    for entry in WalkDir::new("matrices")
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension() == Some("mtx".as_ref()))
    {
        let f = entry.path().file_name().unwrap();
        match parse_matrix_market::<i64, f64>(&fs::read_to_string(entry.path()).unwrap()).unwrap() {
            MatrixType::Integer(_) => {}
            MatrixType::Real(m) => {
                inner!(m, f, c);
            }
            MatrixType::Complex(m) => {
                inner!(m, f, c);
            }
        }
    }
}

criterion_group!(benches, bench_mkl_spmm);
criterion_main!(benches);
