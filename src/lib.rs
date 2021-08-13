#![feature(allocator_api)]
#![deny(clippy::disallowed_method)]
#[cfg(feature = "mkl")]
use mkl_sys::MKL_Complex16;
use num::{Complex, Num, One, Zero};
use std::{
    fmt::Debug,
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};

pub mod arbitrary;
pub mod csr_matrix;
pub mod dok_matrix;
#[cfg(test)]
mod proptest;
#[cfg(test)]
mod tests;

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

fn is_increasing<T: Ord>(s: &[T]) -> bool {
    let mut max = None;
    for i in s {
        if max.map_or(true, |k| k < i) {
            max = Some(i);
        } else {
            return false;
        }
    }
    true
}

fn all_distinct<T: std::hash::Hash + Eq>(s: &[T]) -> bool {
    s.iter().collect::<std::collections::HashSet<_>>().len() == s.len()
}

fn is_sorted<T: Ord>(s: &[T]) -> bool {
    let mut max = None;
    for i in s {
        if max.map_or(true, |k| k <= i) {
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

impl From<Slice> for std::ops::Range<usize> {
    fn from(s: Slice) -> Self {
        s.start..s.start + s.len
    }
}

macro_rules! impl_bin_op {
    ($trait:ident,$op:ident) => {
        impl<T: num::traits::Num + std::clone::Clone> $trait for crate::ComplexNewtype<T> {
            type Output = Self;
            fn $op(self, rhs: Self) -> Self::Output {
                Self(self.0.$op(rhs.0))
            }
        }
    };
}

macro_rules! impl_bin_op_assign {
    ($trait:ident,$op:ident) => {
        impl<T: num::traits::NumAssign + std::clone::Clone> $trait for crate::ComplexNewtype<T> {
            fn $op(&mut self, rhs: Self) {
                self.0.$op(rhs.0);
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComplexNewtype<T>(Complex<T>);

impl_bin_op!(Add, add);
impl_bin_op!(Sub, sub);
impl_bin_op!(Mul, mul);
impl_bin_op!(Div, div);
impl_bin_op!(Rem, rem);

impl_bin_op_assign!(AddAssign, add_assign);
impl_bin_op_assign!(SubAssign, sub_assign);
impl_bin_op_assign!(MulAssign, mul_assign);
impl_bin_op_assign!(DivAssign, div_assign);
impl_bin_op_assign!(RemAssign, rem_assign);

impl<T: Num + Clone> Num for ComplexNewtype<T> {
    type FromStrRadixErr = <Complex<T> as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let z = num::complex::Complex::from_str_radix(str, radix)?;
        Ok(ComplexNewtype(z))
    }
}

impl<T: Num + Clone> Zero for ComplexNewtype<T> {
    fn zero() -> Self {
        ComplexNewtype(Complex::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<T: Num + Clone> One for ComplexNewtype<T> {
    fn one() -> Self {
        ComplexNewtype(Complex::one())
    }
}

#[cfg(feature = "mkl")]
impl From<MKL_Complex16> for ComplexNewtype<f64> {
    fn from(z: MKL_Complex16) -> Self {
        ComplexNewtype(Complex {
            re: z.real,
            im: z.imag,
        })
    }
}

#[cfg(feature = "mkl")]
impl From<ComplexNewtype<f64>> for MKL_Complex16 {
    fn from(z: ComplexNewtype<f64>) -> Self {
        MKL_Complex16 {
            real: z.0.re,
            imag: z.0.im,
        }
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
