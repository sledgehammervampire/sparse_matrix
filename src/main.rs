use num::traits::NumAssign;
use spam::{
    csr_matrix::CsrMatrix,
    dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
};
use std::{ffi::OsStr, fmt::Debug, fs::read_to_string};
use walkdir::WalkDir;

fn main() -> anyhow::Result<()> {
    for entry in WalkDir::new(std::env::args_os().nth(1).unwrap()) {
        let entry = entry?;
        let f = entry.file_name();
        if f == "transient.mtx" {
            match parse_matrix_market::<i64, f64>(&read_to_string(entry.path()).unwrap())
                .unwrap()
                .1
            {
                MatrixType::Integer(m1) => {
                    foo(m1, f);
                }
                MatrixType::Real(m1) => {
                    foo(m1, f);
                }
                MatrixType::Complex(m1) => {
                    foo(m1, f);
                }
            };
        }
    }
    Ok(())
}

fn foo<T: NumAssign + Clone + Send + Sync + Debug>(m1: DokMatrix<T>, _f: &OsStr) {
    let m1 = CsrMatrix::from(m1);
    println!("{:?}", m1.row_nnz_freq());
}
