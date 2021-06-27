use crate::{Matrix, MatrixError};
use itertools::Itertools;
use nom::{Finish, IResult};
use num::Num;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    ops::{Add, Mul},
    str::FromStr,
};
use thiserror::Error;

// a dumb matrix implementation to test against
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct DokMatrix<T> {
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), T>,
}

impl<T: Num> DokMatrix<T> {
    fn new(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        if rows == 0 || cols == 0 {
            return Err(MatrixError::HasZeroDimension);
        }
        Ok(DokMatrix {
            rows,
            cols,
            entries: BTreeMap::new(),
        })
    }

    fn new_square(n: usize) -> Result<Self, MatrixError> {
        Self::new(n, n)
    }

    fn identity(n: usize) -> Result<Self, MatrixError> {
        if n == 0 {
            return Err(MatrixError::HasZeroDimension);
        }
        Ok(DokMatrix {
            rows: n,
            cols: n,
            entries: (0..n).map(|i| ((i, i), T::one())).collect(),
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

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.entries.len()
    }

    // output entries with (row, col) lexicographically ordered
    pub fn entries(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.entries.iter().map(|(&p, t)| (p, t))
    }

    pub fn invariants(&self) -> bool {
        self.rows > 0
            && self.cols > 0
            && self
                .entries()
                .all(|((r, c), t)| r < self.rows && c < self.cols && !t.is_zero())
    }

    fn set_element(&mut self, pos: (usize, usize), t: T) -> Option<T> {
        if t.is_zero() {
            self.entries.remove(&pos)
        } else {
            self.entries.insert(pos, t)
        }
    }
}

impl<T: Num + Clone> Matrix<T> for DokMatrix<T> {
    fn new(rows: usize, cols: usize) -> Result<Self, MatrixError> {
        Self::new(rows, cols)
    }

    fn new_square(n: usize) -> Result<Self, MatrixError> {
        Self::new_square(n)
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn nnz(&self) -> usize {
        self.nnz()
    }

    fn get_element(&self, pos: (usize, usize)) -> Cow<T> {
        self.entries
            .get(&pos)
            .map_or(Cow::Owned(T::zero()), Cow::Borrowed)
    }

    fn set_element(&mut self, pos: (usize, usize), t: T) -> Option<T> {
        self.set_element(pos, t)
    }

    fn identity(n: usize) -> Result<Self, MatrixError> {
        Self::identity(n)
    }

    fn transpose(self) -> Self {
        self.transpose()
    }
}

impl<T> IntoIterator for DokMatrix<T> {
    type Item = ((usize, usize), T);

    type IntoIter = std::collections::btree_map::IntoIter<(usize, usize), T>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<T: Num> Add for DokMatrix<T> {
    type Output = DokMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        DokMatrix {
            rows: self.rows,
            cols: self.cols,
            entries: self
                .entries
                .into_iter()
                .merge_join_by(rhs.entries, |e1, e2| e1.0.cmp(&e2.0))
                .filter_map(|eob| match eob {
                    itertools::EitherOrBoth::Both((p, t1), (_, t2)) => {
                        let t = t1 + t2;
                        if t.is_zero() {
                            None
                        } else {
                            Some((p, t))
                        }
                    }
                    itertools::EitherOrBoth::Left((p, t))
                    | itertools::EitherOrBoth::Right((p, t)) => Some((p, t)),
                })
                .collect(),
        }
    }
}

impl<T: Num + Clone> Mul for &DokMatrix<T> {
    type Output = DokMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut entries = BTreeMap::new();
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut t = T::zero();
                for k in 0..self.cols {
                    t = t + self.get_element((i, k)).into_owned()
                        * rhs.get_element((k, j)).into_owned();
                }
                if !t.is_zero() {
                    entries.insert((i, j), t);
                }
            }
        }

        DokMatrix {
            entries,
            rows: self.rows,
            cols: rhs.cols,
        }
    }
}

impl<T: Num + Clone> From<crate::csr_matrix::CsrMatrix<T>> for DokMatrix<T> {
    fn from(old: crate::csr_matrix::CsrMatrix<T>) -> Self {
        DokMatrix {
            rows: old.rows(),
            cols: old.cols(),
            entries: old.iter().map(|(i, t)| (i, t.clone())).collect(),
        }
    }
}

#[derive(Debug)]
pub enum MatrixType<I, R> {
    Integer(DokMatrix<I>),
    Real(DokMatrix<R>),
    Complex(DokMatrix<num::complex::Complex<R>>),
}

#[derive(Error, Debug)]
pub enum FromMatrixMarketError {
    #[error("parsing error")]
    Nom(#[from] nom::error::Error<String>),
    #[error("{0:?}")]
    MatrixError(#[from] MatrixError),
}

pub fn parse_matrix_market<I: FromStr + Num + Clone, R: FromStr + Num + Clone>(
    input: &str,
) -> Result<MatrixType<I, R>, FromMatrixMarketError> {
    enum EntryType<I, R> {
        Integer(BTreeMap<(usize, usize), I>),
        Real(BTreeMap<(usize, usize), R>),
        Complex(BTreeMap<(usize, usize), num::complex::Complex<R>>),
    }

    fn inner<I: FromStr + Num + Clone, R: FromStr + Num + Clone>(
        input: &str,
    ) -> IResult<&str, (usize, usize, EntryType<I, R>)> {
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
        let entry_type = match entry_type {
            "integer" => Integer(BTreeMap::new()),
            "real" => Real(BTreeMap::new()),
            "complex" => Complex(BTreeMap::new()),
            _ => todo!("entry type {} unsupported", entry_type),
        };
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

        match (entry_type, shape) {
            (Integer(entries), General) => {
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
                    entries,
                    general,
                )(input)?;
                Ok((input, (rows, cols, Integer(entries))))
            }
            (Integer(entries), Symmetric) => {
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
                    entries,
                    symmetric,
                )(input)?;
                Ok((input, (rows, cols, Integer(entries))))
            }
            (Real(entries), General) => {
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
                    entries,
                    general,
                )(input)?;
                Ok((input, (rows, cols, Real(entries))))
            }
            (Real(entries), Symmetric) => {
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
                    entries,
                    symmetric,
                )(input)?;
                Ok((input, (rows, cols, Real(entries))))
            }
            (Complex(entries), General) => {
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
                        |(r, _, c, _, re, _, im, _)| (r, c, num::complex::Complex { re, im }),
                    ),
                    entries,
                    general,
                )(input)?;
                Ok((input, (rows, cols, Complex(entries))))
            }
            (Complex(entries), Symmetric) => {
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
                        |(r, _, c, _, re, _, im, _)| (r, c, num::complex::Complex { re, im }),
                    ),
                    entries,
                    symmetric,
                )(input)?;
                Ok((input, (rows, cols, Complex(entries))))
            }
        }
    }

    let (_, (rows, cols, entries)) =
        inner(input)
            .finish()
            .map_err(|nom::error::Error { input, code }| nom::error::Error {
                input: input.to_string(),
                code,
            })?;
    if rows == 0 || cols == 0 {
        return Err(FromMatrixMarketError::MatrixError(
            MatrixError::HasZeroDimension,
        ));
    }
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

#[cfg(test)]
impl<T: proptest::arbitrary::Arbitrary + Num + Clone> DokMatrix<T> {
    pub fn arb_fixed_size_matrix(
        rows: usize,
        cols: usize,
    ) -> impl proptest::strategy::Strategy<Value = Self> {
        use proptest::prelude::*;
        if rows == 0 || cols == 0 {}
        proptest::collection::btree_map(
            (0..rows, 0..cols),
            T::arbitrary().prop_filter("T is 0", |t| !t.is_zero()),
            0..=(rows * cols),
        )
        .prop_map(move |entries| DokMatrix {
            rows,
            cols,
            entries,
        })
    }

    pub fn arb_matrix() -> impl proptest::strategy::Strategy<Value = Self> {
        crate::proptest::arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}
