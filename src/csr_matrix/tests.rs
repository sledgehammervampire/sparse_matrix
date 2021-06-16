use std::{iter::repeat_with, num::Wrapping};

use hashbag::HashBag;
use itertools::iproduct;
use num::Num;
use proptest::{
    arbitrary::any,
    collection::{self, SizeRange},
    prop_assert, prop_assert_eq,
    sample::subsequence,
    strategy::{Just, Strategy},
    test_runner::TestRunner,
};

use super::{heapsort, insertion_sort, shift_tail, CsrMatrix};
use crate::{
    dok_matrix::DokMatrix,
    proptest::{arb_add_pair, arb_mul_pair},
    AddPair, Matrix, MulPair,
};

const MAX_SIZE: usize = 10;

// base cases
#[test]
fn new_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&(0..MAX_SIZE, 0..MAX_SIZE), |(m, n)| {
            prop_assert!(CsrMatrix::<i8>::new(m, n).invariants());
            Ok(())
        })
        .unwrap();
}

#[test]
fn identity_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&(0..MAX_SIZE), |n| {
            prop_assert!(CsrMatrix::<i8>::identity(n).invariants());
            Ok(())
        })
        .unwrap();
}

#[test]
fn arb_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i8>::arb_matrix(), |m| {
            prop_assert!(m.invariants(), "{:?}", m);
            Ok(())
        })
        .unwrap();
}

#[test]
fn from_arb_dok_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            let m = CsrMatrix::from(m);
            prop_assert!(m.invariants(), "{:?}", m);
            Ok(())
        })
        .unwrap();
}

// inductive cases
#[test]
fn add() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |AddPair(m1, m2)| {
                let m = CsrMatrix::from(m1.clone()) + CsrMatrix::from(m2.clone());
                prop_assert!(m.invariants(), "{:?}", m);
                prop_assert_eq!(m, CsrMatrix::from(m1 + m2));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn mul() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| {
                let m = &CsrMatrix::from(m1.clone()) * &CsrMatrix::from(m2.clone());
                prop_assert!(m.invariants(), "{:?}", m);
                prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn mul_dense() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| {
                let m = CsrMatrix::from(m1.clone()).mul_dense(&CsrMatrix::from(m2.clone()));
                prop_assert!(m.invariants(), "{:?}", m);
                prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn mul_hash() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| {
                let m = CsrMatrix::from(m1.clone()).mul_hash(&CsrMatrix::from(m2.clone()));
                prop_assert!(m.invariants(), "{:?}", m);
                prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn mul_btree() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| {
                let m = CsrMatrix::from(m1.clone()).mul_btree(&CsrMatrix::from(m2.clone()));
                prop_assert!(m.invariants(), "{:?}", m);
                prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn mul_heap() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| {
                let m = CsrMatrix::from(m1.clone()).mul_hash(&CsrMatrix::from(m2.clone()));
                prop_assert!(m.invariants(), "{:?}", m);
                prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn transpose() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            let m1 = CsrMatrix::from(m.clone()).transpose();
            prop_assert!(m1.invariants(), "{:?}", m1);
            prop_assert_eq!(m1, CsrMatrix::from(m.transpose()));
            Ok(())
        })
        .unwrap();
}

#[test]
fn set_element() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &(1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(|(rows, cols)| {
                (
                    DokMatrix::<i8>::arb_fixed_size_matrix(rows, cols),
                    0..rows,
                    0..cols,
                    any::<i8>(),
                )
            }),
            |(mut m, i, j, t)| {
                let mut m1 = CsrMatrix::from(m.clone());
                m.set_element((i, j), t.clone());
                m1.set_element((i, j), t.clone());
                assert!(m1.invariants(), "{:?}", m1);
                assert_eq!(m1, CsrMatrix::from(m));
                Ok(())
            },
        )
        .unwrap();
}

// other
#[test]
fn iter() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            let m1 = CsrMatrix::from(m.clone());
            prop_assert!(m1.iter().eq(m.entries()));
            Ok(())
        })
        .unwrap();
}

#[test]
fn rows_and_cols() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            let m1 = CsrMatrix::from(m.clone());
            prop_assert_eq!((m.rows(), m.cols()), (m1.rows(), m1.cols()));
            Ok(())
        })
        .unwrap();
}

#[test]
fn get_element() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            let m1 = CsrMatrix::from(m.clone());
            prop_assert!(iproduct!(0..m.rows(), 0..m.cols())
                .all(|pos| m.get_element(pos) == m1.get_element(pos)));
            Ok(())
        })
        .unwrap();
}

#[test]
fn convert() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            prop_assert_eq!(&m, &DokMatrix::from(CsrMatrix::from(m.clone())));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_shift_tail() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &collection::vec(any::<bool>(), SizeRange::default()).prop_flat_map(|v| {
                let w = collection::vec(any::<u8>(), v.len()..=v.len());
                (Just(v), w)
            }),
            |(v, w)| {
                prop_assert_eq!(v.len(), w.len());
                let (mut v1, mut w1) = (v.clone(), w.clone());
                unsafe {
                    shift_tail(&mut v1[..], &mut w1[..]);
                }
                if let Some(&x) = v.last() {
                    let i = v1.iter().rposition(|y| *y == x).unwrap();
                    prop_assert_eq!(&v1[..i], &v[..i]);
                    prop_assert_eq!(&w1[..i], &w[..i]);
                    prop_assert_eq!(v1[i], x);
                    prop_assert_eq!(w1[i], *w.last().unwrap());
                    prop_assert_eq!(&v1[i + 1..], &v[i..v.len() - 1]);
                    prop_assert_eq!(&w1[i + 1..], &w[i..w.len() - 1]);
                    prop_assert!(v1[i + 1..].iter().all(|y| *y > x));
                    if i > 0 {
                        prop_assert!(v1[i - 1] <= x);
                    }
                } else {
                    prop_assert_eq!(v, v1);
                    prop_assert_eq!(w, w1);
                }
                Ok(())
            },
        )
        .unwrap();
}

fn test_sort<
    T: Copy + Ord + std::fmt::Debug + proptest::arbitrary::Arbitrary + Eq + std::hash::Hash,
    U: Clone + proptest::arbitrary::Arbitrary + Eq,
>(
    sort: unsafe fn(&mut [T], &mut [U]),
) {
    let mut runner = TestRunner::default();
    runner
        .run(
            &collection::vec(any::<T>(), SizeRange::default()).prop_flat_map(|v| {
                let w = collection::vec(any::<U>(), v.len()..=v.len());
                (Just(v), w)
            }),
            |(v, w)| {
                let (mut v1, mut w1) = (v.clone(), w.clone());
                unsafe {
                    sort(&mut v1[..], &mut w1[..]);
                }
                todo!();
                prop_assert!(v1.is_sorted());
                prop_assert_eq!(
                    v1.iter().copied().collect::<HashBag<_>>(),
                    v.iter().copied().collect::<HashBag<_>>()
                );
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn test_insertion_sort() {
    test_sort::<bool, u8>(insertion_sort);
}

#[test]
fn test_heapsort() {
    test_sort::<bool, u8>(heapsort);
}

impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T> {
    pub fn arb_fixed_size_matrix(rows: usize, cols: usize) -> impl Strategy<Value = CsrMatrix<T>> {
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

    pub fn arb_matrix() -> impl Strategy<Value = Self> {
        crate::proptest::arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}
