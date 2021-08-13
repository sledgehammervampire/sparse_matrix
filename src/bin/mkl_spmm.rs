#![cfg(feature = "mkl")]
use num::Num;
use open_ambient::open_ambient_dir;
use spam::{csr_matrix::{
        mkl::{MklCsrMatrix, RustMklSparseMatrix},
        CsrMatrix,
    }, dok_matrix::DokMatrix, gen_mul_main};
use std::convert::TryFrom;

const ITERS: usize = 100;

gen_mul_main!(mkl_spmm);

fn main() -> anyhow::Result<()> {
    let dir = open_ambient_dir!("matrices")?;
    mul_main(dir)?;
    Ok(())
}

fn mkl_spmm<T, U>(m: DokMatrix<T>)
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
