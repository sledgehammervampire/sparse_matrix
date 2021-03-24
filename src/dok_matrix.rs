use crate::{arbitrary::arb_matrix, csr_matrix::CsrMatrix, Matrix};
use itertools::Itertools;
use nom::IResult;
use num::{traits::NumAssignRef, Num};
use proptest::{arbitrary::Arbitrary, prelude::*, strategy::Strategy};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    ops::{Add, Mul},
    str::FromStr,
};

// a dumb matrix implementation to test against
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct DokMatrix<T> {
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), T>,
}

impl<T: Num + Clone> DokMatrix<T> {
    // output entries with (row, col) lexicographically ordered
    pub fn entries(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.entries.iter().map(|(&p, t)| (p, t))
    }
}

impl<T: Arbitrary + Num> DokMatrix<T> {
    pub fn arb_fixed_size_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Self> {
        prop::collection::btree_map(
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

    pub fn arb_matrix() -> impl Strategy<Value = Self> {
        arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

impl<T: Num + Clone> Matrix<T> for DokMatrix<T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn get_element(&self, pos: (usize, usize)) -> Cow<T> {
        self.entries
            .get(&pos)
            .map_or(Cow::Owned(T::zero()), Cow::Borrowed)
    }

    fn set_element(&mut self, pos: (usize, usize), t: T) {
        self.entries.insert(pos, t);
    }

    fn identity(n: usize) -> Self {
        DokMatrix {
            rows: n,
            cols: n,
            entries: (0..n).map(|i| ((i, i), T::one())).collect(),
        }
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

impl<T: NumAssignRef + Clone> Mul for &DokMatrix<T> {
    type Output = DokMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut entries = BTreeMap::new();
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let t = (0..self.cols).fold(T::zero(), |mut s, k| {
                    let mut t = self.entries.get(&(i, k)).unwrap_or(&T::zero()).clone();
                    t *= rhs.entries.get(&(k, j)).unwrap_or(&T::zero());
                    s += t;
                    s
                });
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

impl<T: Num + Clone> From<CsrMatrix<T>> for DokMatrix<T> {
    fn from(old: CsrMatrix<T>) -> Self {
        DokMatrix {
            rows: old.rows(),
            cols: old.cols(),
            entries: old.iter().map(|(i, t)| (i, t.clone())).collect(),
        }
    }
}

pub fn parse_matrix_market<T: FromStr + Clone>(input: &str) -> IResult<&str, DokMatrix<T>> {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{char, digit1, line_ending, not_line_ending},
        combinator::{map, map_res, opt, recognize},
        multi::{fold_many0, many0},
        sequence::{delimited, pair, preceded, tuple},
    };

    fn parse_num<T: FromStr>(input: &str) -> IResult<&str, T> {
        map_res(recognize(pair(opt(char('-')), digit1)), str::parse)(input)
    }

    fn matrix_size(input: &str) -> IResult<&str, (usize, usize)> {
        map(
            tuple((
                parse_num::<usize>,
                char(' '),
                parse_num::<usize>,
                char(' '),
                parse_num::<usize>,
                line_ending,
            )),
            |(r, _, c, _, _, _)| (r, c),
        )(input)
    }

    fn matrix_entry<T: FromStr>(input: &str) -> IResult<&str, (usize, usize, T)> {
        map(
            tuple((
                parse_num::<usize>,
                char(' '),
                parse_num::<usize>,
                char(' '),
                parse_num::<T>,
                line_ending,
            )),
            |(r, _, c, _, e, _)| (r, c, e),
        )(input)
    }

    // parse header
    let (input, _) = tag("%%MatrixMarket matrix coordinate")(input)?;
    let (input, entry_type) = preceded(
        char(' '),
        alt((tag("integer"), tag("real"), tag("complex"), tag("pattern"))),
    )(input)?;
    if entry_type != "integer" {
        unimplemented!("matrix entry type {} unsupported", entry_type);
    }
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
    // parse comments
    let (input, _) = many0(delimited(char('%'), not_line_ending, line_ending))(input)?;
    let (input, (rows, cols)) = matrix_size(input)?;
    match shape {
        "symmetric" => {
            let (input, entries) = fold_many0(
                matrix_entry::<T>,
                BTreeMap::new(),
                |mut entries, (r, c, t)| {
                    // matrix market format is 1-indexed, but our matrix is 0-indexed
                    entries.insert((r - 1, c - 1), t.clone());
                    entries.insert((c - 1, r - 1), t);
                    entries
                },
            )(input)?;
            Ok((
                input,
                DokMatrix {
                    rows,
                    cols,
                    entries,
                },
            ))
        }
        "general" => {
            let (input, entries) = fold_many0(
                matrix_entry::<T>,
                BTreeMap::new(),
                |mut entries, (r, c, t)| {
                    // matrix market format is 1-indexed, but our matrix is 0-indexed
                    entries.insert((r - 1, c - 1), t.clone());
                    entries
                },
            )(input)?;
            Ok((
                input,
                DokMatrix {
                    rows,
                    cols,
                    entries,
                },
            ))
        }
        _ => {
            unimplemented!("matrix shape {} unsupported", shape);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DokMatrix;
    use crate::{csr_matrix::CsrMatrix, Matrix};

    #[test]
    fn test_1() {
        assert_eq!(
            DokMatrix::from(CsrMatrix::identity(2)),
            DokMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect()
            }
        )
    }

    #[test]
    fn test_2() {
        assert_eq!(
            CsrMatrix::from({
                let entries = vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect();
                DokMatrix {
                    rows: 2,
                    cols: 2,
                    entries,
                }
            }),
            CsrMatrix::identity(2)
        );
    }
    #[test]
    fn test_3() {
        assert_eq!(
            CsrMatrix::identity(3) + CsrMatrix::identity(3),
            CsrMatrix::from(DokMatrix {
                rows: 3,
                cols: 3,
                entries: vec![((0, 0), 2), ((1, 1), 2), ((2, 2), 2)]
                    .into_iter()
                    .collect()
            })
        )
    }

    #[test]
    fn test_4() {
        assert_eq!(
            {
                let mut m = CsrMatrix::identity(2);
                m.set_element((0, 1), 1);
                m
            },
            CsrMatrix::from(DokMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                    .into_iter()
                    .collect()
            })
        )
    }

    #[test]
    fn test_5() {
        assert_eq!(
            {
                let mut m = CsrMatrix::from(DokMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                        .into_iter()
                        .collect(),
                });
                m.set_element((0, 1), 0);
                m
            },
            CsrMatrix::identity(2)
        )
    }

    #[test]
    fn test_6() {
        assert_eq!(
            {
                let mut m = CsrMatrix::from(DokMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                        .into_iter()
                        .collect(),
                });
                m.set_element((0, 1), 2);
                m
            },
            CsrMatrix::from(DokMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 2), ((1, 1), 1)]
                    .into_iter()
                    .collect(),
            })
        )
    }

    #[test]
    fn test_7() {
        assert_eq!(
            {
                let mut m = CsrMatrix::from(DokMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect(),
                });
                m.set_element((0, 1), 0);
                m
            },
            CsrMatrix::identity(2)
        )
    }

    #[test]
    fn test_8() {
        assert_eq!(
            {
                let mut m = CsrMatrix::identity(2);
                m.set_element((0, 0), 0);
                m.set_element((1, 1), 0);
                m
            },
            CsrMatrix::from(DokMatrix {
                rows: 2,
                cols: 2,
                entries: vec![].into_iter().collect()
            })
        )
    }

    #[test]
    fn test_9() {
        assert_eq!(
            &CsrMatrix::<i32>::identity(2) * &CsrMatrix::identity(2),
            CsrMatrix::identity(2)
        )
    }

    #[test]
    fn test_10() {
        assert_eq!(
            {
                let m = CsrMatrix::from(DokMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![((0, 1), 1)].into_iter().collect(),
                });
                &m * &m.clone()
            },
            {
                CsrMatrix::from(DokMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![].into_iter().collect(),
                })
            }
        )
    }

    #[test]
    fn test_11() {
        assert_eq!(
            CsrMatrix::<i32>::identity(3),
            CsrMatrix::identity(3).transpose()
        )
    }

    #[test]
    fn test_12() {
        assert_eq!(
            CsrMatrix::from(DokMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 1), 1)].into_iter().collect()
            })
            .transpose(),
            CsrMatrix::from(DokMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((1, 0), 1)].into_iter().collect()
            })
        )
    }
}
