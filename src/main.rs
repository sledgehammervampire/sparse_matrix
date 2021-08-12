use cap_std::fs::Dir;
use num::{traits::NumAssign, Num};
use open_ambient::open_ambient_dir;
use spam::{
    csr_matrix::{
        mkl::{MklCsrMatrix, RustMklSparseMatrix},
        CsrMatrix,
    },
    dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
};
use std::{convert::TryFrom, io::Read};

const ITERS: usize = 100;

fn main() -> anyhow::Result<()> {
    let dir = open_ambient_dir!("matrices")?;
    bench_mul(dir)?;
    Ok(())
}

fn bench_mul(dir: Dir) -> anyhow::Result<()> {
    for entry in dir.entries()? {
        let entry = entry?;
        let mut input = String::new();
        entry.open()?.read_to_string(&mut input)?;
        match parse_matrix_market::<i64, f64>(&input).unwrap() {
            MatrixType::Integer(_) => {}
            MatrixType::Real(m) => {
                mul_hash(m);
            }
            MatrixType::Complex(m) => {
                mul_hash(m);
            }
        }
    }
    Ok(())
}

pub fn mkl_spmm<T, U>(m: DokMatrix<T>)
where
    T: Num,
    MklCsrMatrix<U, true>: TryFrom<CsrMatrix<T, true>>,
    for<'a> RustMklSparseMatrix<'a, U, true>: TryFrom<&'a mut MklCsrMatrix<U, true>>,
{
    let m = CsrMatrix::from(m);
    if let Ok(mut m) = MklCsrMatrix::try_from(m) {
        if let Ok(m) = RustMklSparseMatrix::try_from(&mut m) {
            for _ in 0..ITERS {
                let _ = &m * &m;
            }
        }
    }
}

pub fn mul_hash<T: NumAssign + Copy + Send + Sync>(m: DokMatrix<T>) {
    let m = CsrMatrix::from(m);
    for _ in 0..ITERS {
        let _: CsrMatrix<_, false> = m.mul_hash(&m);
    }
}
