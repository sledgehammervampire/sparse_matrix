use num::Num;
use open_ambient::open_ambient_dir;
use spam_csr::{
    gen_mul_main,
    mkl::{IntoMklScalar, MklCsrMatrix, MklScalar, RustMklSparseMatrix},
    CsrMatrix,
};
use spam_dok::DokMatrix;
use std::convert::TryFrom;

const ITERS: usize = 100;

gen_mul_main!(mkl_spmm);

pub fn main() -> anyhow::Result<()> {
    let dir = open_ambient_dir!("../matrices")?;
    mul_main(dir)?;
    Ok(())
}

fn mkl_spmm<T>(m: DokMatrix<T>)
where
    T: Num + IntoMklScalar,
    <T as IntoMklScalar>::Output: MklScalar,
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
