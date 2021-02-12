use std::{
    collections::BTreeMap,
    ops::{AddAssign, Mul},
};

use num::{traits::NumAssignRef, Num};
use proptest::prelude::*;

use crate::CsrMatrix;

// a dumb matrix implementation to test against
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct DokMatrix<T> {
    rows: usize,
    cols: usize,
    entries: BTreeMap<(usize, usize), T>,
}
const MAX_SIZE: usize = 100;

impl<T> DokMatrix<T> {
    pub fn entries(&self) -> &BTreeMap<(usize, usize), T> {
        &self.entries
    }
}

impl<T: Num + Clone> From<DokMatrix<T>> for CsrMatrix<T> {
    fn from(old: DokMatrix<T>) -> Self {
        let mut new = CsrMatrix::new(old.rows, old.cols);
        for ((i, j), t) in old.entries {
            new.set_element((i, j), t);
        }
        new
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

impl<T: NumAssignRef + Clone> AddAssign<&DokMatrix<T>> for DokMatrix<T> {
    fn add_assign(&mut self, rhs: &DokMatrix<T>) {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );

        for (&(i, j), t) in &rhs.entries {
            let entry = self.entries.entry((i, j)).or_insert(T::zero());
            *entry += t;
            // TODO: check if entry == 0
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

pub fn arb_matrix_with_rows_and_cols<T: Arbitrary>(
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
    (1..MAX_SIZE, 1..MAX_SIZE)
        .prop_flat_map(|(rows, cols)| arb_matrix_with_rows_and_cols(rows, cols))
}

// pair of matrices conformable for addition
#[derive(Clone, Debug)]
pub struct AddPair<T>(pub DokMatrix<T>, pub DokMatrix<T>);

pub fn arb_add_pair<T: Arbitrary + Clone>() -> impl Strategy<Value = AddPair<T>> {
    arb_matrix().prop_flat_map(|m| {
        arb_matrix_with_rows_and_cols(m.rows, m.cols).prop_map(move |m1| AddPair(m.clone(), m1))
    })
}
/*
// pair of matrices conformable for multiplication
#[derive(Clone, Debug)]
pub struct MulPair<T>(pub DokMatrix<T>, pub DokMatrix<T>);

impl<T: Arbitrary + Num> Arbitrary for MulPair<T> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        let m = DokMatrix::arbitrary(u)?;
        let (rows, cols) = (m.cols, u.int_in_range(1..=MAX_SIZE)?);
        Ok(MulPair(m, arbitrary_matrix(u, rows, cols)?))
    }
}
*/
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
            {
                let mut m1 = CsrMatrix::identity(3);
                let m2 = CsrMatrix::identity(3);
                m1 += &m2;
                m1
            },
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
    /*
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
    */
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
