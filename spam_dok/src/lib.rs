#![feature(type_alias_impl_trait)]
#![cfg_attr(test, feature(no_coverage))]
#![deny(clippy::disallowed_method)]

use cmplx::ComplexNewtype;
use conv::prelude::*;
use itertools::Itertools;
use nom::{Finish, IResult};
use num::{traits::NumAssign, Num};
#[cfg(feature = "proptest-arbitrary")]
use proptest::prelude::*;
#[cfg(test)]
use serde::{Deserialize, Serialize};
#[cfg(feature = "proptest-arbitrary")]
use spam_matrix::proptest::arb_matrix;
use spam_matrix::{IndexError, Matrix};
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    ops::{Add, Mul, Sub},
    str::FromStr,
};
use thiserror::Error;

#[cfg(test)]
mod tests;

// a dumb matrix implementation to test against
#[cfg_attr(test, derive(Deserialize, Serialize))]
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct DokMatrix<T> {
    rows: NonZeroUsize,
    cols: NonZeroUsize,
    entries: BTreeMap<(usize, usize), T>,
}

impl DokMatrix<f64> {
    pub fn debug_in_scientific_notation(&self) -> String {
        let mut s = String::from("{");
        s.push_str(
            &self
                .iter()
                .map(|((i, j), t)| format!("({}, {}, {:e})", i, j, t))
                .join(", "),
        );
        s.push_str("}");
        s
    }
}

#[derive(Debug)]
pub struct IsNan;
impl DokMatrix<f64> {
    // see (3.13) from Accuracy and stability of numerical algorithms by Higham
    pub fn good_matrix_approx(&self, rhs: &Self, approx: &Self) -> Result<bool, IsNan> {
        let inf_norm = |m: &DokMatrix<f64>| {
            let mut max = 0.0;
            for rsum in (0..m.rows().get()).map(|r| {
                m.entries
                    .range((r, 0)..(r + 1, 0))
                    .map(|(_, t)| t.abs())
                    .sum::<f64>()
            }) {
                if rsum.is_nan() {
                    return Err(IsNan);
                } else if rsum > max {
                    max = rsum;
                }
            }
            Ok(max)
        };
        let n = f64::value_from(self.rows().get().max(self.cols().get())).unwrap();
        let u = f64::EPSILON / 2.0;
        let gamma = n * u / (1.0 - n * u);
        let expected = self * rhs;
        if expected.iter().all(|(_, t)| !t.is_nan()) && approx.iter().any(|(_, t)| t.is_nan()) {
            Ok(false)
        } else {
            let self_norm = inf_norm(self)?;
            let rhs_norm = inf_norm(rhs)?;
            Ok(inf_norm(&(expected - approx.clone()))?
                <= 2.0
                    * gamma
                    * if self_norm == 0.0 || rhs_norm == 0.0 {
                        // don't want 0.0*inf to become NaN
                        0.0
                    } else {
                        self_norm * rhs_norm
                    })
        }
    }
}

impl<T: Num> DokMatrix<T> {
    // output entries with (row, col) lexicographically ordered
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.entries.iter().map(|(&p, t)| (p, t))
    }

    fn apply_elementwise<F>(self, rhs: Self, f: &F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        use itertools::EitherOrBoth::*;
        let mut m = DokMatrix::new((self.rows(), self.cols()));
        self.entries
            .into_iter()
            .merge_join_by(rhs.entries, |e1, e2| e1.0.cmp(&e2.0))
            .map(|eob| match eob {
                Both((p, t1), (_, t2)) => (p, f(t1, t2)),
                Left((p, t)) => (p, f(t, T::zero())),
                Right((p, t)) => (p, f(T::zero(), t)),
            })
            .for_each(|(pos, t)| {
                m.set_element(pos, t).unwrap();
            });
        m
    }
}

impl<T: Num> Matrix<T> for DokMatrix<T> {
    fn invariants(&self) -> bool {
        self.iter()
            .all(|((r, c), t)| r < self.rows.get() && c < self.cols.get() && !t.is_zero())
    }

    fn new((rows, cols): (NonZeroUsize, NonZeroUsize)) -> Self {
        DokMatrix {
            rows,
            cols,
            entries: BTreeMap::new(),
        }
    }

    fn new_square(n: NonZeroUsize) -> Self {
        Self::new((n, n))
    }

    fn identity(n: NonZeroUsize) -> Self {
        DokMatrix {
            rows: n,
            cols: n,
            entries: (0..n.get()).map(|i| ((i, i), T::one())).collect(),
        }
    }

    fn rows(&self) -> NonZeroUsize {
        self.rows
    }

    fn cols(&self) -> NonZeroUsize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.entries.len()
    }

    fn get_element(&self, (i, j): (usize, usize)) -> Result<Option<&T>, IndexError> {
        if !(i < self.rows.get() && j < self.cols.get()) {
            return Err(IndexError);
        }
        Ok(self.entries.get(&(i, j)))
    }

    fn set_element(&mut self, (i, j): (usize, usize), t: T) -> Result<Option<T>, IndexError> {
        if !(i < self.rows.get() && j < self.cols.get()) {
            return Err(IndexError);
        }
        Ok(if t.is_zero() {
            self.entries.remove(&(i, j))
        } else {
            self.entries.insert((i, j), t)
        })
    }

    fn transpose(self) -> Self {
        DokMatrix {
            rows: self.cols,
            cols: self.rows,
            entries: self
                .entries
                .into_iter()
                .map(|((i, j), t)| ((j, i), t))
                .collect(),
        }
    }
}

impl<T: Num> Add for DokMatrix<T> {
    type Output = DokMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::add)
    }
}
impl<T: Num> Sub for DokMatrix<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::sub)
    }
}

impl<T: Num + Clone> Mul for &DokMatrix<T> {
    type Output = DokMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut m = DokMatrix::new((self.rows(), rhs.cols()));
        for i in 0..self.rows.get() {
            for j in 0..rhs.cols.get() {
                let mut t = T::zero();
                for k in 0..self.cols.get() {
                    t = t + self
                        .get_element((i, k))
                        .unwrap()
                        .cloned()
                        .unwrap_or_else(T::zero)
                        * rhs
                            .get_element((k, j))
                            .unwrap()
                            .cloned()
                            .unwrap_or_else(T::zero);
                }
                m.set_element((i, j), t).unwrap();
            }
        }
        m
    }
}
impl<T> IntoIterator for DokMatrix<T> {
    type Item = ((usize, usize), T);

    type IntoIter = impl Iterator<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

#[cfg(feature = "proptest-arbitrary")]
impl<T: Arbitrary + Num + Clone> DokMatrix<T> {
    pub fn arb_fixed_size_matrix(
        rows: NonZeroUsize,
        cols: NonZeroUsize,
    ) -> impl Strategy<Value = Self> {
        proptest::collection::vec(
            ((0..rows.get(), 0..cols.get()), T::arbitrary()),
            0..=(2 * rows.get() * cols.get()),
        )
        .prop_map(move |entries| {
            let mut m = DokMatrix::new((rows, cols));
            for (pos, t) in entries {
                m.set_element(pos, t).unwrap();
            }
            m
        })
    }

    pub fn arb_matrix() -> impl Strategy<Value = Self> {
        arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

pub enum MatrixType<I, F: NumAssign + Clone> {
    Integer(DokMatrix<I>),
    Real(DokMatrix<F>),
    Complex(DokMatrix<ComplexNewtype<F>>),
}

#[derive(Error, Debug)]
pub enum FromMatrixMarketError {
    #[error("parsing error")]
    Nom(#[from] nom::error::Error<String>),
    #[error("number of rows or columns is 0")]
    HasZeroDimension,
}

pub fn parse_matrix_market<I: FromStr + num::Integer + Clone, F: FromStr + NumAssign + Clone>(
    input: &str,
) -> Result<MatrixType<I, F>, FromMatrixMarketError> {
    enum EntryType<I, F: NumAssign + Clone> {
        Integer(BTreeMap<(usize, usize), I>),
        Real(BTreeMap<(usize, usize), F>),
        Complex(BTreeMap<(usize, usize), ComplexNewtype<F>>),
    }

    fn inner<I: FromStr + Num + Clone, F: FromStr + NumAssign + Clone>(
        input: &str,
    ) -> IResult<&str, (usize, usize, EntryType<I, F>)> {
        use nom::{
            branch::alt,
            bytes::complete::tag,
            character::complete::{char, digit1, line_ending, not_line_ending},
            combinator::{map, map_res, opt, recognize},
            multi::{fold_many0, many0},
            number::complete::recognize_float,
            sequence::{delimited, pair, preceded, tuple},
        };
        use EntryType::*;
        use MatrixShape::*;

        enum MatrixShape {
            General,
            Symmetric,
        }

        fn recognize_int<T: FromStr>(input: &str) -> IResult<&str, &str> {
            recognize(pair(opt(char('-')), digit1))(input)
        }
        fn parse_usize(input: &str) -> IResult<&str, usize> {
            map_res(recognize_int::<usize>, str::parse)(input)
        }
        fn matrix_size(input: &str) -> IResult<&str, (usize, usize)> {
            map(
                tuple({
                    (
                        parse_usize,
                        char(' '),
                        parse_usize,
                        char(' '),
                        parse_usize,
                        line_ending,
                    )
                }),
                |(r, _, c, _, _, _)| (r, c),
            )(input)
        }
        fn general<T: Num>(
            mut entries: BTreeMap<(usize, usize), T>,
            (r, c, t): (usize, usize, T),
        ) -> BTreeMap<(usize, usize), T> {
            if !t.is_zero() {
                // matrix market format is 1-indexed, but our matrix is 0-indexed
                entries.insert((r - 1, c - 1), t);
            }
            entries
        }
        fn symmetric<T: Num + Clone>(
            mut entries: BTreeMap<(usize, usize), T>,
            (r, c, t): (usize, usize, T),
        ) -> BTreeMap<(usize, usize), T> {
            if !t.is_zero() {
                // matrix market format is 1-indexed, but our matrix is 0-indexed
                entries.insert((r - 1, c - 1), t.clone());
                entries.insert((c - 1, r - 1), t);
            }
            entries
        }

        // parse header
        let (input, _) = tag("%%MatrixMarket matrix coordinate")(input)?;
        let (input, entry_type) = preceded(
            char(' '),
            alt((tag("integer"), tag("real"), tag("complex"), tag("pattern"))),
        )(input)?;
        let (input, shape) = delimited(
            char(' '),
            alt((
                tag("general"),
                tag("symmetric"),
                tag("skew-symmetric"),
                tag("hermitian"),
            )),
            line_ending,
        )(input)?;
        let shape = match shape {
            "general" => General,
            "symmetric" => Symmetric,
            _ => todo!("matrix shape {} unsupported", shape),
        };
        // parse comments
        let (input, _) = many0(delimited(char('%'), not_line_ending, line_ending))(input)?;
        let (input, (rows, cols)) = matrix_size(input)?;

        match entry_type {
            "integer" => {
                let (input, entries) = fold_many0(
                    map(
                        tuple({
                            (
                                parse_usize,
                                char(' '),
                                parse_usize,
                                char(' '),
                                map_res(recognize_int::<I>, str::parse),
                                line_ending,
                            )
                        }),
                        |(r, _, c, _, e, _)| (r, c, e),
                    ),
                    BTreeMap::new,
                    match shape {
                        General => general,
                        Symmetric => symmetric,
                    },
                )(input)?;
                Ok((input, (rows, cols, Integer(entries))))
            }
            "real" => {
                let (input, entries) = fold_many0(
                    map(
                        tuple({
                            (
                                parse_usize,
                                char(' '),
                                parse_usize,
                                char(' '),
                                map_res(recognize_float, str::parse),
                                line_ending,
                            )
                        }),
                        |(r, _, c, _, e, _)| (r, c, e),
                    ),
                    BTreeMap::new,
                    match shape {
                        General => general,
                        Symmetric => symmetric,
                    },
                )(input)?;
                Ok((input, (rows, cols, Real(entries))))
            }
            "complex" => {
                let (input, entries) = fold_many0(
                    map(
                        tuple((
                            parse_usize,
                            char(' '),
                            parse_usize,
                            char(' '),
                            map_res(recognize_float, str::parse),
                            char(' '),
                            map_res(recognize_float, str::parse),
                            line_ending,
                        )),
                        |(r, _, c, _, re, _, im, _)| {
                            (r, c, ComplexNewtype(num::complex::Complex { re, im }))
                        },
                    ),
                    BTreeMap::new,
                    general,
                )(input)?;
                Ok((input, (rows, cols, Complex(entries))))
            }
            _ => todo!("entry type {} unsupported", entry_type),
        }
    }

    let (_, (rows, cols, entries)) =
        inner(input)
            .finish()
            .map_err(|nom::error::Error { input, code }| nom::error::Error {
                input: input.to_string(),
                code,
            })?;
    let (rows, cols) = (
        NonZeroUsize::new(rows).ok_or(FromMatrixMarketError::HasZeroDimension)?,
        NonZeroUsize::new(cols).ok_or(FromMatrixMarketError::HasZeroDimension)?,
    );
    match entries {
        EntryType::Integer(entries) => Ok(MatrixType::Integer(DokMatrix {
            rows,
            cols,
            entries,
        })),
        EntryType::Real(entries) => Ok(MatrixType::Real(DokMatrix {
            rows,
            cols,
            entries,
        })),
        EntryType::Complex(entries) => Ok(MatrixType::Complex(DokMatrix {
            rows,
            cols,
            entries,
        })),
    }
}

pub fn into_float_matrix_market<F: num::Float + std::fmt::Display, W: std::fmt::Write>(
    m: DokMatrix<F>,
    w: &mut W,
) -> Result<(), std::fmt::Error> {
    writeln!(w, "%%MatrixMarket matrix coordinate real general")?;
    writeln!(w, "{} {} {}", m.rows(), m.cols(), m.nnz())?;
    for ((i, j), t) in m.entries {
        writeln!(w, "{} {} {}", i + 1, j + 1, t)?;
    }
    Ok(())
}
