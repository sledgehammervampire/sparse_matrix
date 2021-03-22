use crate::{CsrMatrix, Matrix};
use itertools::Itertools;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{digit1, line_ending, not_line_ending},
    combinator::{map, map_res, opt, recognize},
    multi::{fold_many0, many0},
    sequence::{delimited, pair, tuple},
    IResult,
};
use num::{traits::NumAssignRef, Num};
use proptest::{arbitrary::Arbitrary, prelude::*, sample::subsequence, strategy::Strategy};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    iter::repeat_with,
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
}

impl<T: Num + Clone> From<DokMatrix<T>> for CsrMatrix<T> {
    fn from(old: DokMatrix<T>) -> Self {
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);
        for ((i, j), t) in old.entries {
            vals.push(t);
            cidx.push(j);
            if let None = ridx.get(i + 1) {
                let &k = ridx.last().unwrap();
                for _ in ridx.len()..=i + 1 {
                    ridx.push(k);
                }
            }
            ridx[i + 1] += 1;
        }
        let &k = ridx.last().unwrap();
        for _ in ridx.len()..=old.rows {
            ridx.push(k);
        }
        CsrMatrix {
            rows: old.rows,
            cols: old.cols,
            vals,
            cidx,
            ridx,
        }
    }
}

impl<T: Num + Clone> From<CsrMatrix<T>> for DokMatrix<T> {
    fn from(old: CsrMatrix<T>) -> Self {
        DokMatrix {
            rows: old.rows,
            cols: old.cols,
            entries: old.iter().map(|(i, t)| (i, t.clone())).collect(),
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

const MAX_SIZE: usize = 100;

pub fn arb_dok_matrix_fixed_size<T: Arbitrary + Num>(
    rows: usize,
    cols: usize,
) -> impl Strategy<Value = DokMatrix<T>> {
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

pub fn arb_csr_matrix_fixed_size<T: Arbitrary + Num>(
    rows: usize,
    cols: usize,
) -> impl Strategy<Value = CsrMatrix<T>> {
    repeat_with(|| subsequence((0..cols).collect::<Vec<_>>(), 0..=cols))
        .take(rows)
        .collect::<Vec<_>>()
        .prop_flat_map(move |cidx| {
            let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
            for mut rcidx in cidx {
                ridx.push(ridx.last().unwrap() + rcidx.len());
                cidx_flattened.append(&mut rcidx);
            }
            repeat_with(|| T::arbitrary().prop_filter("T is 0", |t| !t.is_zero()))
                .take(cidx_flattened.len())
                .collect::<Vec<_>>()
                .prop_map(move |vals| CsrMatrix {
                    rows,
                    cols,
                    vals,
                    cidx: cidx_flattened.clone(),
                    ridx: ridx.clone(),
                })
        })
}

fn arb_matrix<T: Arbitrary, F: Fn(usize, usize) -> S, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = S::Value> {
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(move |(rows, cols)| arb_matrix_fixed_size(rows, cols))
}

pub fn arb_dok_matrix<T: Arbitrary + Num>() -> impl Strategy<Value = DokMatrix<T>> {
    arb_matrix::<T, _, _>(arb_dok_matrix_fixed_size)
}

pub fn arb_csr_matrix<T: Arbitrary + Num>() -> impl Strategy<Value = CsrMatrix<T>> {
    arb_matrix::<T, _, _>(arb_csr_matrix_fixed_size)
}

// pair of matrices conformable for addition
#[derive(Clone, Debug)]
pub struct AddPair<M>(pub M, pub M);

pub fn arb_add_pair_fixed_size<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S, S: Strategy>(
    rows: usize,
    cols: usize,
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = AddPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    arb_matrix_fixed_size(rows, cols).prop_flat_map(move |m| {
        arb_matrix_fixed_size(m.rows(), m.cols()).prop_map(move |m1| AddPair(m.clone(), m1))
    })
}

pub fn arb_add_pair<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S + Copy, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = AddPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(move |(rows, cols)| {
        arb_add_pair_fixed_size(rows, cols, arb_matrix_fixed_size)
    })
}

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<M>(pub M, pub M);

pub fn arb_mul_pair_fixed_size<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S, S: Strategy>(
    l: usize,
    n: usize,
    p: usize,
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = MulPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    arb_matrix_fixed_size(l, n).prop_flat_map(move |m| {
        arb_matrix_fixed_size(n, p).prop_map(move |m1| MulPair(m.clone(), m1))
    })
}

pub fn arb_mul_pair<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S + Copy, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = MulPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    (1..MAX_SIZE, 1..MAX_SIZE, 1..MAX_SIZE)
        .prop_flat_map(move |(l, n, p)| arb_mul_pair_fixed_size(l, n, p, arb_matrix_fixed_size))
}

fn parse_num<T: FromStr>(input: &str) -> IResult<&str, T> {
    use nom::character::complete::char;

    map_res(recognize(pair(opt(char('-')), digit1)), str::parse)(input)
}

fn matrix_size(input: &str) -> IResult<&str, (usize, usize)> {
    use nom::character::complete::char;

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
    use nom::character::complete::char;

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

pub fn parse_matrix_market<T: FromStr + Clone>(input: &str) -> IResult<&str, DokMatrix<T>> {
    use nom::character::complete::char;

    // parse header
    let (input, _) = tag("%%MatrixMarket matrix coordinate integer")(input)?;
    let (input, shape) = delimited(
        char(' '),
        alt((tag("general"), tag("symmetric"))),
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
                    entries.insert((r-1, c-1), t.clone());
                    entries.insert((c-1, r-1), t);
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
                    entries.insert((r-1, c-1), t.clone());
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
            todo!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DokMatrix;
    use crate::CsrMatrix;

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
