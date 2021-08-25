use cap_std::fs::Dir;
use criterion::Criterion;
use open_ambient::open_ambient_dir;
use spam_csr::{
    mkl::{MklCsrMatrix, RustMklSparseMatrix},
    CsrMatrix,
};
use spam_dok::{parse_matrix_market, MatrixType};
use std::{convert::TryFrom, io::Read};

pub fn main() -> anyhow::Result<()> {
    let dir = open_ambient_dir!("../matrices")?;
    bench_mul(dir)?;
    Ok(())
}

fn bench_mul(dir: Dir) -> anyhow::Result<()> {
    let mut criterion = Criterion::default().configure_from_args();
    for entry in dir.entries()? {
        let entry = entry?;
        let mut input = String::new();
        entry.open()?.read_to_string(&mut input)?;
        match parse_matrix_market::<i64, f64>(&input).unwrap() {
            MatrixType::Integer(_) => {}
            MatrixType::Real(m) => {
                let m = CsrMatrix::from(m);
                let mut m = MklCsrMatrix::try_from(m).unwrap();
                let m = RustMklSparseMatrix::try_from(&mut m).unwrap();
                criterion.bench_function(&format!("bench mkl_spmm {:?}", entry.file_name()), |b| {
                    b.iter(|| {
                        let _ = &m * &m;
                    });
                });
            }
            MatrixType::Complex(m) => {
                let m = CsrMatrix::from(m);
                let mut m = MklCsrMatrix::try_from(m).unwrap();
                let m = RustMklSparseMatrix::try_from(&mut m).unwrap();
                criterion.bench_function(&format!("bench mkl_spmm {:?}", entry.file_name()), |b| {
                    b.iter(|| {
                        let _ = &m * &m;
                    });
                });
            }
        }
    }
    criterion.final_summary();
    Ok(())
}
