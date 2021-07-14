use mkl_sys::MKL_Complex16;
use num::{Complex, Num, One, Zero};
use std::{
    borrow::Cow,
    collections::HashSet,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};
use thiserror::Error;

pub mod arbitrary;
pub mod csr_matrix;
pub mod dok_matrix;
#[cfg(feature = "mkl")]
pub mod mkl;
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
    s.iter().collect::<HashSet<_>>().len() == s.len()
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

impl From<MKL_Complex16> for ComplexNewtype<f64> {
    fn from(z: MKL_Complex16) -> Self {
        ComplexNewtype(Complex {
            re: z.real,
            im: z.imag,
        })
    }
}

impl From<ComplexNewtype<f64>> for MKL_Complex16 {
    fn from(z: ComplexNewtype<f64>) -> Self {
        MKL_Complex16 {
            real: z.0.re,
            imag: z.0.im,
        }
    }
}

#[macro_export]
macro_rules! gen_bench {
    ($func:ident, $sorted:literal) => {
        fn benches() {
            let mut criterion = Criterion::default().configure_from_args();
            for m in vec!["cont-300", "crystk02", "matrix-new_3", "nemeth20"] {
                bench(&mut criterion, m);
            }

            fn bench(c: &mut Criterion, matrix_name: &str) {
                use spam::{
                    csr_matrix::CsrMatrix,
                    dok_matrix::{parse_matrix_market, MatrixType},
                };
                use std::{fs, path::Path};

                let path = format!("matrices/{}.mtx", matrix_name);
                let path = Path::new(&path);
                match parse_matrix_market::<i64, f64>(&fs::read_to_string(path).unwrap()).unwrap() {
                    MatrixType::Integer(m) => {
                        let m = CsrMatrix::<_, $sorted>::from(m);
                        c.bench_function(&format!("bench {} {:?}", stringify!($func), path), |b| {
                            b.iter(|| {
                                let _: CsrMatrix<_, $sorted> = m.$func(&m);
                            });
                        });
                    }
                    MatrixType::Real(m) => {
                        let m = CsrMatrix::<_, $sorted>::from(m);
                        c.bench_function(&format!("bench {} {:?}", stringify!($func), path), |b| {
                            b.iter(|| {
                                let _: CsrMatrix<_, $sorted> = m.$func(&m);
                            });
                        });
                    }
                    MatrixType::Complex(m) => {
                        let m = CsrMatrix::<_, $sorted>::from(m);
                        c.bench_function(&format!("bench {} {:?}", stringify!($func), path), |b| {
                            b.iter(|| {
                                let _: CsrMatrix<_, $sorted> = m.$func(&m);
                            });
                        });
                    }
                }
            }
        }
    };
}
