use std::{fs::read_to_string, num::Wrapping};

use num::Num;
use proptest::{prop_assert, prop_assert_eq, strategy::Strategy, test_runner::TestRunner};

use super::{parse_matrix_market, DokMatrix};
use crate::{
    csr_matrix::CsrMatrix,
    proptest::{arb_add_pair, arb_matrix, arb_mul_pair},
    AddPair, MulPair,
};

const MAX_SIZE: usize = 10;

impl<T: proptest::arbitrary::Arbitrary + Num> DokMatrix<T> {
    pub fn arb_fixed_size_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Self> {
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

    pub fn arb_matrix() -> impl Strategy<Value = Self> {
        arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

// base cases
#[test]
fn identity_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&(1..MAX_SIZE), |n| {
            prop_assert!(DokMatrix::<i8>::identity(n).invariants());
            Ok(())
        })
        .unwrap();
}

#[test]
fn arb_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            prop_assert!(m.invariants());
            Ok(())
        })
        .unwrap();
}

#[test]
fn from_arb_csr_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i8>::arb_matrix(), |m| {
            let m = DokMatrix::from(m);
            prop_assert!(m.invariants());
            Ok(())
        })
        .unwrap();
}

#[ignore = "expensive, files don't change"]
#[test]
fn from_matrix_market() {
    for f in &["sc2010", "tube2", "big", "gr_30_30", "bcsstm01"] {
        let m = parse_matrix_market::<i32>(&read_to_string(format!("matrices/{}.mtx", f)).unwrap())
            .unwrap()
            .1;
        assert!(m.invariants());
    }
}

// inductive cases
#[test]
fn add() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |AddPair(m1, m2)| {
                let m = m1.clone() + m2.clone();
                prop_assert!(m.invariants());
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
                let m = &m1 * &m2;
                prop_assert!(m.invariants());
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
            let m1 = m.clone().transpose();
            prop_assert!(m1.invariants());
            Ok(())
        })
        .unwrap();
}

#[test]
fn set_element() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_add_pair(DokMatrix::<i8>::arb_fixed_size_matrix),
            |AddPair(mut m1, m2)| {
                for (pos, t) in m2.entries() {
                    m1.set_element(pos, t.clone());
                }
                prop_assert!(m1.invariants());
                Ok(())
            },
        )
        .unwrap();
}

// other
#[test]
fn convert() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i8>::arb_matrix(), |m| {
            prop_assert_eq!(&m, &CsrMatrix::from(DokMatrix::from(m.clone())));
            Ok(())
        })
        .unwrap();
}
