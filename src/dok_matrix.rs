use crate::{csr_matrix::CsrMatrix, ComplexNewtype, IndexError, Matrix};
use itertools::Itertools;
use nom::{Finish, IResult};
use num::Num;
use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    ops::{Add, Mul},
    str::FromStr,
};
use thiserror::Error;

// a dumb matrix implementation to test against
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct DokMatrix<T> {
    rows: NonZeroUsize,
    cols: NonZeroUsize,
    pub(crate) entries: BTreeMap<(usize, usize), T>,
}

impl<T: Num> DokMatrix<T> {
    // output entries with (row, col) lexicographically ordered
    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.entries.iter().map(|(&p, t)| (p, t))
    }

    pub fn invariants(&self) -> bool {
        self.iter()
            .all(|((r, c), t)| r < self.rows.get() && c < self.cols.get() && !t.is_zero())
    }
}

impl<T: Num + Clone> Matrix<T> for DokMatrix<T> {
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

impl<T: Num, const IS_SORTED: bool> From<CsrMatrix<T, IS_SORTED>> for DokMatrix<T> {
    fn from(old: CsrMatrix<T, IS_SORTED>) -> Self {
        DokMatrix {
            rows: old.rows(),
            cols: old.cols(),
            entries: old
                .into_iter()
                .filter_map(|(i, t)| if t.is_zero() { None } else { Some((i, t)) })
                .collect(),
        }
    }
}

pub enum MatrixType<I, R> {
    Integer(DokMatrix<I>),
    Real(DokMatrix<R>),
    Complex(DokMatrix<ComplexNewtype<R>>),
}

#[derive(Error, Debug)]
pub enum FromMatrixMarketError {
    #[error("parsing error")]
    Nom(#[from] nom::error::Error<String>),
    #[error("number of rows or columns is 0")]
    HasZeroDimension,
}

pub fn parse_matrix_market<I: FromStr + Num + Clone, R: FromStr + Num + Clone>(
    input: &str,
) -> Result<MatrixType<I, R>, FromMatrixMarketError> {
    enum EntryType<I, R> {
        Integer(BTreeMap<(usize, usize), I>),
        Real(BTreeMap<(usize, usize), R>),
        Complex(BTreeMap<(usize, usize), ComplexNewtype<R>>),
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
                        |(r, _, c, _, re, _, im, _)| {
                            (r, c, ComplexNewtype(num::complex::Complex { re, im }))
                        },
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
                        |(r, _, c, _, re, _, im, _)| {
                            (r, c, ComplexNewtype(num::complex::Complex { re, im }))
                        },
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
