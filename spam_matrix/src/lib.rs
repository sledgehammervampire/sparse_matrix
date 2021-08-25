#![cfg_attr(test, feature(no_coverage, type_alias_impl_trait))]
#![feature(allocator_api, is_sorted)]
#![deny(clippy::disallowed_method)]

use std::{fmt::Debug, num::NonZeroUsize};
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
#[cfg(feature = "proptest")]
pub mod proptest;

#[derive(Debug, PartialEq, Eq)]
pub struct IndexError;

pub trait Matrix<T>: Sized {
    fn invariants(&self) -> bool;
    fn new(size: (NonZeroUsize, NonZeroUsize)) -> Self;
    fn new_square(n: NonZeroUsize) -> Self;
    fn identity(n: NonZeroUsize) -> Self;
    fn rows(&self) -> NonZeroUsize;
    fn cols(&self) -> NonZeroUsize;
    // the number of explicit entries in the matrix
    fn nnz(&self) -> usize;
    fn get_element(&self, pos: (usize, usize)) -> Result<Option<&T>, IndexError>;
    fn set_element(&mut self, pos: (usize, usize), t: T) -> Result<Option<T>, IndexError>;
    fn transpose(self) -> Self;
}

// pair of matrices conformable for addition
#[derive(Clone, Debug)]
pub struct AddPair<M>(pub M, pub M);

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<M>(pub M, pub M);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Slice {
    pub start: usize,
    pub len: usize,
}

impl From<Slice> for std::ops::Range<usize> {
    fn from(s: Slice) -> Self {
        s.start..s.start + s.len
    }
}

#[macro_export]
macro_rules! gen_bench_mul {
    ($f:ident) => {
        fn bench_mul<const OUTPUT_SORTED: bool>(dir: cap_std::fs::Dir) -> anyhow::Result<()> {
            use cap_std::fs::DirEntry;
            use criterion::Criterion;
            use num::traits::NumAssign;
            use spam::{
                csr_matrix::CsrMatrix,
                dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
            };
            use std::io::Read;

            fn inner<T: NumAssign + Send + Sync + Copy, const B: bool>(
                m: DokMatrix<T>,
                criterion: &mut Criterion,
                entry: DirEntry,
            ) {
                let m = CsrMatrix::from(m);
                criterion.bench_function(
                    &format!("bench {} {:?}", stringify!($f), entry.file_name()),
                    |b| {
                        b.iter(|| {
                            let _: CsrMatrix<_, B> = m.$f(&m);
                        });
                    },
                );
            }

            let mut criterion = Criterion::default().configure_from_args();
            for entry in dir.entries()? {
                let entry = entry?;
                let mut input = String::new();
                entry.open()?.read_to_string(&mut input)?;
                match parse_matrix_market::<i64, f64>(&input).unwrap() {
                    MatrixType::Integer(m) => {
                        inner::<_, OUTPUT_SORTED>(m, &mut criterion, entry);
                    }
                    MatrixType::Real(m) => {
                        inner::<_, OUTPUT_SORTED>(m, &mut criterion, entry);
                    }
                    MatrixType::Complex(m) => {
                        inner::<_, OUTPUT_SORTED>(m, &mut criterion, entry);
                    }
                }
            }
            criterion.final_summary();
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! gen_mul_main {
    ($f:ident) => {
        fn mul_main(dir: cap_std::fs::Dir) -> anyhow::Result<()> {
            use spam::dok_matrix::{parse_matrix_market, MatrixType};
            use std::io::Read;

            for entry in dir.entries()? {
                let entry = entry?;
                let mut input = String::new();
                entry.open()?.read_to_string(&mut input)?;
                match parse_matrix_market::<i64, f64>(&input).unwrap() {
                    MatrixType::Integer(_) => {}
                    MatrixType::Real(m) => {
                        $f(m);
                    }
                    MatrixType::Complex(m) => {
                        $f(m);
                    }
                }
            }
            Ok(())
        }
    };
}
