use num::traits::NumAssign;
use open_ambient::open_ambient_dir;
use spam_csr::{CsrMatrix, gen_mul_main};
use spam_dok::DokMatrix;

const ITERS: usize = 100;

gen_mul_main!(mul_hash);

fn main() -> anyhow::Result<()> {
    let dir = open_ambient_dir!("../matrices")?;
    mul_main(dir)?;
    Ok(())
}

pub fn mul_hash<T: NumAssign + Copy + Send + Sync>(m: DokMatrix<T>) {
    let m = CsrMatrix::from(m);
    for _ in 0..ITERS {
        let _: CsrMatrix<_, false> = m.mul_hash(&m);
    }
}
