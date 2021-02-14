use std::{
    collections::BTreeMap,
    ops::{Add, Mul},
};

use itertools::Itertools;
use num::{traits::NumAssignRef, Num};
use proptest::prelude::*;

use crate::CsrMatrix;

const MAX_SIZE: usize = 100;

// a dumb matrix implementation to test against
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct DokMatrix<T> {
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), T>,
}

impl<T> DokMatrix<T> {
    // output entries with (row, col) lexicographically ordered
    pub fn entries(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.entries.iter().map(|(&p, t)| (p, t))
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
                .map(|eob| match eob {
                    itertools::EitherOrBoth::Both((p, t1), (_, t2)) => (p, t1 + t2),
                    itertools::EitherOrBoth::Left((p, t))
                    | itertools::EitherOrBoth::Right((p, t)) => (p, t),
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

pub fn arb_matrix_fixed_size<T: Arbitrary>(
    rows: usize,
    cols: usize,
) -> impl Strategy<Value = DokMatrix<T>> {
    proptest::collection::btree_map((0..rows, 0..cols), T::arbitrary(), 0..=(rows * cols / 10))
        .prop_map(move |entries| DokMatrix {
            rows,
            cols,
            entries,
        })
}

pub fn arb_matrix<T: Arbitrary>() -> impl Strategy<Value = DokMatrix<T>> {
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(|(rows, cols)| arb_matrix_fixed_size(rows, cols))
}

// pair of matrices conformable for addition
#[derive(Clone, Debug)]
pub struct AddPair<T>(pub DokMatrix<T>, pub DokMatrix<T>);

pub fn arb_add_pair_fixed_size<T: Arbitrary + Clone>(
    rows: usize,
    cols: usize,
) -> impl Strategy<Value = AddPair<T>> {
    arb_matrix_fixed_size(rows, cols).prop_flat_map(|m| {
        arb_matrix_fixed_size(m.rows, m.cols).prop_map(move |m1| AddPair(m.clone(), m1))
    })
}

pub fn arb_add_pair<T: Arbitrary + Clone>() -> impl Strategy<Value = AddPair<T>> {
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(|(rows, cols)| arb_add_pair_fixed_size(rows, cols))
}

// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<T>(pub DokMatrix<T>, pub DokMatrix<T>);

pub fn arb_mul_pair_fixed_size<T: Arbitrary + Clone>(
    l: usize,
    n: usize,
    p: usize,
) -> impl Strategy<Value = MulPair<T>> {
    arb_matrix_fixed_size(l, n).prop_flat_map(move |m| {
        arb_matrix_fixed_size(n, p).prop_map(move |m1| MulPair(m.clone(), m1))
    })
}

pub fn arb_mul_pair<T: Arbitrary + Clone>() -> impl Strategy<Value = MulPair<T>> {
    (1..MAX_SIZE, 1..MAX_SIZE, 1..MAX_SIZE)
        .prop_flat_map(|(l, n, p)| arb_mul_pair_fixed_size(l, n, p))
}

#[cfg(test)]
mod test {
    use crate::CsrMatrix;

    use super::DokMatrix;

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
