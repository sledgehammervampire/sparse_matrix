use cap_rand::prelude::CapRng;
use cap_std::{ambient_authority, fs::Dir};
use criterion::Criterion;
use std::io::Read;

use spam::{
    csr_matrix::CsrMatrix,
    dok_matrix::{parse_matrix_market, MatrixType},
};

fn main() -> anyhow::Result<()> {
    let ambient_authority = ambient_authority();
    let dir = Dir::open_ambient_dir("matrices", ambient_authority)?;
    let mut rng = CapRng::default(ambient_authority);
    bench_mul(dir, &mut rng)?;

    Ok(())
}

fn bench_mul(dir: cap_std::fs::Dir, rng: &mut CapRng) -> anyhow::Result<()> {
    let mut criterion = Criterion::default().configure_from_args();
    for entry in dir.entries()? {
        let entry = entry?;
        let mut input = String::new();
        entry.open()?.read_to_string(&mut input)?;
        match parse_matrix_market::<i64, f64>(&input).unwrap() {
            MatrixType::Integer(m) => {
                let m = CsrMatrix::from_dok(m, rng);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!(mul_hash), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, false> = m.mul_hash(&m);
                        });
                    },
                );
            }
            MatrixType::Real(m) => {
                let m = CsrMatrix::from_dok(m, rng);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!(mul_hash), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, false> = m.mul_hash(&m);
                        });
                    },
                );
            }
            MatrixType::Complex(m) => {
                let m = CsrMatrix::from_dok(m, rng);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!(mul_hash), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, false> = m.mul_hash(&m);
                        });
                    },
                );
            }
        }
    }
    criterion.final_summary();
    Ok(())
}
