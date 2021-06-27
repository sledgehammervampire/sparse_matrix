use std::{borrow::Cow, ops::Range};
use thiserror::Error;

pub mod arbitrary;
pub mod csr_matrix;
pub mod dok_matrix;
#[cfg(test)]
mod proptest;
#[cfg(test)]
mod tests;

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("number of rows is 0 or number of columns is 0")]
    HasZeroDimension,
}

pub trait Matrix<T: ToOwned>: Sized {
    fn new(rows: usize, cols: usize) -> Result<Self, MatrixError>;
    fn new_square(n: usize) -> Result<Self, MatrixError>;
    fn identity(n: usize) -> Result<Self, MatrixError>;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    // the number of nonzero entries in the matrix
    fn nnz(&self) -> usize;
    fn get_element(&self, pos: (usize, usize)) -> Cow<T>;
    fn set_element(&mut self, pos: (usize, usize), t: T) -> Option<T>;
    fn transpose(self) -> Self;
}

// pair of matrices conformable for addition
#[derive(Clone, Debug)]
pub struct AddPair<M>(pub M, pub M);

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<M>(pub M, pub M);

fn is_increasing(s: &[usize]) -> bool {
    let mut max = None;
    for i in s {
        if Some(i) > max {
            max = Some(i);
        } else {
            return false;
        }
    }
    true
}

fn is_sorted(s: &[usize]) -> bool {
    let mut max = None;
    for i in s {
        if Some(i) >= max {
            max = Some(i);
        } else {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Slice {
    pub start: usize,
    pub len: usize,
}

impl From<Slice> for Range<usize> {
    fn from(s: Slice) -> Self {
        s.start..s.start + s.len
    }
}

#[macro_export]
macro_rules! make_bench_mul {
    ($bench_name:ident, $func_name:ident) => {
        pub fn $bench_name(c: &mut criterion::Criterion) {
            use criterion::Criterion;
            use num::traits::NumAssign;
            use spam::{
                csr_matrix::CsrMatrix,
                dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
                Matrix,
            };
            use std::{ffi::OsStr, fs};
            use walkdir::WalkDir;

            fn inner<T: Clone + NumAssign + Send + Sync>(
                c: &mut Criterion,
                f: &OsStr,
                m: DokMatrix<T>,
            ) {
                let m = CsrMatrix::from(m);
                c.bench_function(
                    &format!(
                        "bench {:?} {:?} ({}x{}, {} nonzero entries)",
                        stringify!($func_name),
                        f,
                        m.rows(),
                        m.cols(),
                        m.nnz()
                    ),
                    |b| b.iter(|| m.$func_name(&m)),
                );
            }

            for entry in WalkDir::new("matrices")
                .into_iter()
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.path().extension() == Some("mtx".as_ref()))
            {
                let f = entry.path().file_name().unwrap();
                match parse_matrix_market::<i64, f64>(&fs::read_to_string(entry.path()).unwrap())
                    .unwrap()
                {
                    MatrixType::Integer(m) => {
                        inner(c, f, m);
                    }
                    MatrixType::Real(m) => {
                        inner(c, f, m);
                    }
                    MatrixType::Complex(m) => {
                        inner(c, f, m);
                    }
                }
            }
        }
    };
}
