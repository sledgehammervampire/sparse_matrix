use cap_std::fs::Dir;
use criterion::Criterion;
use open_ambient::open_ambient_dir;
use std::io::Read;

use spam::{
    csr_matrix::CsrMatrix,
    dok_matrix::{parse_matrix_market, MatrixType},
};

fn main() -> anyhow::Result<()> {
    let dir = open_ambient_dir!("matrices")?;
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
            MatrixType::Integer(m) => {
                let m = CsrMatrix::from(m);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!(mul_hash), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, false> = m.mul_hash2(&m);
                        });
                    },
                );
            }
            MatrixType::Real(m) => {
                let m = CsrMatrix::from(m);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!(mul_hash), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, false> = m.mul_hash2(&m);
                        });
                    },
                );
            }
            MatrixType::Complex(m) => {
                let m = CsrMatrix::from(m);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!(mul_hash), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, false> = m.mul_hash2(&m);
                        });
                    },
                );
            }
        }
    }
    criterion.final_summary();
    Ok(())
}
