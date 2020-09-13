use crate::{CSRMatrix, Slice};
use num::{traits::NumAssignRef, Num};
use quickcheck::Arbitrary;
use rand::prelude::*;
use std::{
    collections::HashMap,
    iter::repeat_with,
    ops::{AddAssign, MulAssign},
};

#[derive(Debug, Eq, PartialEq, Clone)]
struct DOKMatrix<T> {
    rows: usize,
    cols: usize,
    entries: HashMap<(usize, usize), T>,
}

impl<T: Num + Clone> From<DOKMatrix<T>> for CSRMatrix<T> {
    fn from(old: DOKMatrix<T>) -> Self {
        let mut new = CSRMatrix::new(old.rows, old.cols);
        for ((i, j), t) in old.entries {
            new.set_element((i, j), t);
        }
        new
    }
}
impl<T: Num + Clone> From<CSRMatrix<T>> for DOKMatrix<T> {
    fn from(old: CSRMatrix<T>) -> Self {
        let mut vals = old.vals.into_iter().map(Some).collect::<Vec<_>>();
        let mut entries = HashMap::new();
        for (r, Slice { start, len }) in old.ridx {
            for (&c, t) in old.cidx[start..start + len]
                .iter()
                .zip(vals[start..start + len].iter_mut())
            {
                entries.insert((r, c), t.take().unwrap());
            }
        }

        DOKMatrix {
            rows: old.rows,
            cols: old.cols,
            entries,
        }
    }
}

impl<T: NumAssignRef + Clone> AddAssign<&DOKMatrix<T>> for DOKMatrix<T> {
    fn add_assign(&mut self, rhs: &DOKMatrix<T>) {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );

        for (&(i, j), t) in &rhs.entries {
            let entry = self.entries.entry((i, j)).or_insert(T::zero());
            *entry += t;
        }
    }
}

impl<T: NumAssignRef + Clone> MulAssign<&DOKMatrix<T>> for DOKMatrix<T> {
    fn mul_assign(&mut self, rhs: &DOKMatrix<T>) {
        assert_eq!(self.cols, rhs.rows, "LHS cols != RHS rows");

        let mut entries = HashMap::new();
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

        self.cols = rhs.cols;
        self.entries = entries;
    }
}

fn nonzero_arb<G: quickcheck::Gen, T: Arbitrary + Num>(g: &mut G) -> T {
    repeat_with(|| T::arbitrary(g))
        .skip_while(|t| t.is_zero())
        .next()
        .unwrap()
}

impl<T: Arbitrary + NumAssignRef> Arbitrary for DOKMatrix<T> {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let rows = nonzero_arb(g);
        let cols = nonzero_arb(g);
        let mut entries = HashMap::new();
        for _ in 0..usize::arbitrary(g) {
            entries.insert((g.gen_range(0, rows), g.gen_range(0, cols)), nonzero_arb(g));
        }
        DOKMatrix {
            rows,
            cols,
            entries,
        }
    }
}

#[derive(Clone, Debug)]
struct AddPair<T>(DOKMatrix<T>, DOKMatrix<T>);

impl<T: Arbitrary + NumAssignRef> Arbitrary for AddPair<T> {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let m1 = DOKMatrix::arbitrary(g);
        let (rows, cols) = (m1.rows, m1.cols);
        let n = usize::arbitrary(g);
        let entries =
            repeat_with(|| ((g.gen_range(0, rows), g.gen_range(0, cols)), nonzero_arb(g)))
                .take(n)
                .collect();
        AddPair(
            m1,
            DOKMatrix {
                rows,
                cols,
                entries,
            },
        )
    }
}

#[derive(Clone, Debug)]
struct MulPair<T>(DOKMatrix<T>, DOKMatrix<T>);

impl<T: Arbitrary + NumAssignRef> Arbitrary for MulPair<T> {
    fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
        let m1 = DOKMatrix::arbitrary(g);
        let (rows, cols) = (m1.cols, nonzero_arb(g));
        let n = usize::arbitrary(g);
        let entries =
            repeat_with(|| ((g.gen_range(0, rows), g.gen_range(0, cols)), nonzero_arb(g)))
                .take(n)
                .collect();
        MulPair(
            m1,
            DOKMatrix {
                rows,
                cols,
                entries,
            },
        )
    }
}

#[quickcheck]
fn test_convert(m: DOKMatrix<i32>) -> bool {
    m == DOKMatrix::from(CSRMatrix::from(m.clone()))
}
#[quickcheck]
fn test_add(AddPair(mut m1, m2): AddPair<i32>) -> bool {
    let mut m3 = CSRMatrix::from(m1.clone());
    let m4 = CSRMatrix::from(m2.clone());
    m3 += &m4;
    m1 += &m2;
    CSRMatrix::from(m1) == m3
}
#[quickcheck]
fn test_mul(MulPair(mut m1, m2): MulPair<i32>) -> bool {
    let mut m3 = CSRMatrix::from(m1.clone());
    let m4 = CSRMatrix::from(m2.clone());
    m3 *= &m4;
    m1 *= &m2;
    CSRMatrix::from(m1) == m3
}

#[test]
fn test_1() {
    assert_eq!(
        DOKMatrix::from(CSRMatrix::identity(2)),
        DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect()
        }
    )
}
#[test]
fn test_2() {
    assert_eq!(
        CSRMatrix::from({
            let entries = vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect();
            DOKMatrix {
                rows: 2,
                cols: 2,
                entries,
            }
        }),
        CSRMatrix::identity(2)
    );
}
#[test]
fn test_3() {
    assert_eq!(
        {
            let mut m1 = CSRMatrix::identity(3);
            let m2 = CSRMatrix::identity(3);
            m1 += &m2;
            m1
        },
        CSRMatrix::from(DOKMatrix {
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
            let mut m = CSRMatrix::identity(2);
            m.set_element((0, 1), 1);
            m
        },
        CSRMatrix::from(DOKMatrix {
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
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                    .into_iter()
                    .collect(),
            });
            m.set_element((0, 1), 0);
            m
        },
        CSRMatrix::identity(2)
    )
}
#[test]
fn test_6() {
    assert_eq!(
        {
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                    .into_iter()
                    .collect(),
            });
            m.set_element((0, 1), 2);
            m
        },
        CSRMatrix::from(DOKMatrix {
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
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect(),
            });
            m.set_element((0, 1), 0);
            m
        },
        CSRMatrix::identity(2)
    )
}
#[test]
fn test_8() {
    assert_eq!(
        {
            let mut m = CSRMatrix::identity(2);
            m.set_element((0, 0), 0);
            m.set_element((1, 1), 0);
            m
        },
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![].into_iter().collect()
        })
    )
}
#[test]
fn test_9() {
    assert_eq!(
        {
            let mut m = CSRMatrix::<i32>::identity(2);
            m *= &CSRMatrix::identity(2);
            m
        },
        CSRMatrix::identity(2)
    )
}
#[test]
fn test_10() {
    assert_eq!(
        {
            let mut m = CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 1), 1)].into_iter().collect(),
            });
            m *= &m.clone();
            m
        },
        {
            CSRMatrix::from(DOKMatrix {
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
        CSRMatrix::<i32>::identity(3),
        CSRMatrix::identity(3).transpose()
    )
}
#[test]
fn test_12() {
    assert_eq!(
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((0, 1), 1)].into_iter().collect()
        }).transpose(),
        CSRMatrix::from(DOKMatrix {
            rows: 2,
            cols: 2,
            entries: vec![((1, 0), 1)].into_iter().collect()
        })
    )
}
