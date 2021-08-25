use crate::{into_float_matrix_market, parse_matrix_market, DokMatrix, MatrixType};
use num::Num;
use proptest::{prelude::*, test_runner::TestRunner};
use spam_matrix::{
    proptest::{arb_add_pair, arb_mul_pair},
    AddPair, Matrix, MulPair,
};
use std::{convert::TryInto, num::Wrapping};

const MAX_SIZE: usize = 10;

fn test_invariants<S, F, T>(strategy: &S, f: F)
where
    S: Strategy,
    T: Num,
    F: Fn(S::Value) -> DokMatrix<T>,
{
    let mut runner = TestRunner::default();
    runner
        .run(strategy, |v| {
            let m = f(v);
            prop_assert!(m.invariants());
            Ok(())
        })
        .unwrap();
}

// base cases
#[test]
fn new_invariants() {
    test_invariants(
        &(1..MAX_SIZE, 1..MAX_SIZE)
            .prop_map(|(m, n)| (m.try_into().unwrap(), n.try_into().unwrap())),
        DokMatrix::<i8>::new,
    );
}

#[test]
fn identity_invariants() {
    test_invariants(
        &(1..MAX_SIZE).prop_map(|n| n.try_into().unwrap()),
        DokMatrix::<i8>::identity,
    );
}

#[test]
fn arb_invariants() {
    test_invariants(&DokMatrix::<i8>::arb_matrix(), |m| m);
}

#[test]
fn test_into_float_matrix_market() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<f64>::arb_matrix(), |m| {
            let mut ser = String::new();
            into_float_matrix_market(m.clone(), &mut ser).unwrap();
            if let MatrixType::Real(m1) = parse_matrix_market::<i64, f64>(&ser).unwrap() {
                assert_eq!(m, m1);
            } else {
                unreachable!();
            }
            Ok(())
        })
        .unwrap();
}

// inductive cases
#[test]
fn add() {
    test_invariants(
        &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
        |AddPair(m1, m2)| m1 + m2,
    );
}

#[test]
fn mul() {
    test_invariants(
        &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
        |MulPair(m1, m2)| &m1 * &m2,
    );
}

#[test]
fn transpose() {
    test_invariants(&DokMatrix::<i8>::arb_matrix(), DokMatrix::transpose);
}

#[test]
fn set_element() {
    test_invariants(
        &(1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(|(rows, cols)| {
            (
                DokMatrix::<i8>::arb_fixed_size_matrix(
                    rows.try_into().unwrap(),
                    cols.try_into().unwrap(),
                ),
                0..rows,
                0..cols,
                any::<i8>(),
            )
        }),
        |(mut m, i, j, t)| {
            let t_old = m.get_element((i, j)).unwrap().cloned();
            let t1_old = m.set_element((i, j), t).unwrap();
            assert_eq!(t_old, t1_old);
            m
        },
    );
}
