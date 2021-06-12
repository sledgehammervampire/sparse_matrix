use criterion::{criterion_group, criterion_main};

pub fn bench_mul_dense(c: &mut criterion::Criterion) {
    use criterion::Criterion;
    use num::traits::NumAssign;
    use spam::{
        csr_matrix::CsrMatrix,
        dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
        Matrix,
    };
    use std::{ffi::OsStr, fs};
    use walkdir::WalkDir;

    fn inner<T: Clone + NumAssign + Send + Sync>(c: &mut Criterion, f: &OsStr, m: DokMatrix<T>) {
        let m = CsrMatrix::from(m);
        c.bench_function(
            &format!(
                "bench {:?} {:?} ({}x{}, {} nonzero entries)",
                "mul_dense",
                f,
                m.rows(),
                m.cols(),
                m.nnz()
            ),
            |b| b.iter(|| m.mul_dense(&m)),
        );
    }

    for entry in WalkDir::new("matrices")
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension() == Some("mtx".as_ref()))
        .take(1)
    {
        let f = entry.path().file_name().unwrap();
        match parse_matrix_market::<i64, f64>(&fs::read_to_string(entry.path()).unwrap())
            .unwrap()
            .1
        {
            MatrixType::Integer(m) => {
                inner(c, f, m);
            }
            MatrixType::Real(m) => {
                inner(c, f, m);
            }
            MatrixType::Complex(m) => {
                inner(c, f, m);
            }
        }
    }
}

criterion_group!(benches, bench_mul_dense);
criterion_main!(benches);
