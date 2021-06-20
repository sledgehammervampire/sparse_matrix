use num::traits::NumAssign;
use spam::{
    csr_matrix::{ffi::MklSparseMatrix, CsrMatrix},
    dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
};
use std::{convert::TryFrom, ffi::OsStr, fmt::Debug, fs::read_to_string};
use walkdir::WalkDir;

fn main() -> anyhow::Result<()> {
    for entry in WalkDir::new(std::env::args_os().nth(1).unwrap()) {
        let entry = entry?;
        let f = entry.file_name();
        if let Some(ext) = entry.path().extension() {
            if ext == "mtx" {
                match parse_matrix_market::<i64, f64>(&read_to_string(entry.path()).unwrap())
                    .unwrap()
                    .1
                {
                    MatrixType::Integer(m1) => {
                        foo(m1, f);
                    }
                    MatrixType::Real(m1) => {
                        let m = MklSparseMatrix::try_from(CsrMatrix::from(m1)).unwrap();
                    }
                    MatrixType::Complex(m1) => {
                        foo(m1, f);
                    }
                }
            }
        }
    }
    Ok(())
}

fn foo<T: NumAssign + Clone + Send + Sync + Debug>(m1: DokMatrix<T>, _f: &OsStr) {}
