use cap_std::fs::Dir;
use open_ambient::open_ambient_dir;
use spam::{
    csr_matrix::CsrMatrix,
    dok_matrix::{parse_matrix_market, MatrixType},
};
use std::io::Read;

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
            MatrixType::Integer(m) => {
                let m = CsrMatrix::from(m);
                for _ in 0..100 {
                    let _: CsrMatrix<_, false> = m.mul_hash2(&m);
                }
            }
            MatrixType::Real(m) => {
                let m = CsrMatrix::from(m);
                for _ in 0..100 {
                    let _: CsrMatrix<_, false> = m.mul_hash2(&m);
                }
            }
            MatrixType::Complex(m) => {
                let m = CsrMatrix::from(m);
                for _ in 0..100 {
                    let _: CsrMatrix<_, false> = m.mul_hash2(&m);
                }
            }
        }
    }
    Ok(())
}
