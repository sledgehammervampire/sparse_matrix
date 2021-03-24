use itertools::iproduct;
use proptest::{
    arbitrary::any,
    strategy::{Just, Strategy},
    test_runner::TestRunner,
};

use crate::{
    arbitrary::{arb_add_pair, arb_mul_pair, AddPair, MulPair},
    csr_matrix::invariants::csr_invariants,
    csr_matrix::CsrMatrix,
    dok_matrix::DokMatrix,
    Matrix,
};

#[test]
fn test_csr_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i32>::arb_matrix(), |m| {
            assert!(csr_invariants(&m));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_dok_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i32>::arb_matrix(), |m| {
            let m = CsrMatrix::from(m);
            assert!(csr_invariants(&m));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_set_element_1() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &CsrMatrix::<i32>::arb_matrix()
                .prop_flat_map(|m| (0..m.rows(), 0..m.cols(), Just(m), any::<i32>())),
            |(i, j, mut m, t)| {
                m.set_element((i, j), t);
                assert!(csr_invariants(&m));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn test_set_element_2() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &DokMatrix::<i32>::arb_matrix().prop_flat_map(|m| {
                (
                    0..m.rows(),
                    0..m.cols(),
                    Just(CsrMatrix::from(m)),
                    any::<i32>(),
                )
            }),
            |(i, j, mut m, t)| {
                m.set_element((i, j), t);
                assert!(csr_invariants(&m));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn test_set_element_3() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &DokMatrix::<i32>::arb_matrix()
                .prop_flat_map(|m| (0..m.rows(), 0..m.cols(), Just(m), any::<i32>())),
            |(i, j, mut m, t)| {
                let mut m1 = CsrMatrix::from(m.clone());
                m1.set_element((i, j), t);
                m.set_element((i, j), t);
                assert_eq!(m1, CsrMatrix::from(m));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn test_convert_1() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i32>::arb_matrix(), |m| {
            assert_eq!(m, DokMatrix::from(CsrMatrix::from(m.clone())));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_convert_2() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i32>::arb_matrix(), |m| {
            let m1 = CsrMatrix::from(m.clone());
            assert_eq!((m.rows(), m.cols()), (m1.rows(), m1.cols()));
            assert!(iproduct!(0..m.rows(), 0..m.cols())
                .all(|pos| m.get_element(pos) == m1.get_element(pos)));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_convert_3() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i32>::arb_matrix(), |m| {
            assert_eq!(m, CsrMatrix::from(DokMatrix::from(m.clone())));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_convert_4() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i32>::arb_matrix(), |m| {
            let m1 = DokMatrix::from(m.clone());
            assert!(iproduct!(0..m.rows(), 0..m.cols())
                .all(|pos| m.get_element(pos) == m1.get_element(pos)));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_transpose() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i32>::arb_matrix(), |m| {
            let m1 = m.clone().transpose();
            assert!(iproduct!(0..m.rows(), 0..m.cols())
                .all(|(i, j)| m.get_element((i, j)) == m1.get_element((j, i))));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_iter() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i32>::arb_matrix(), |m| {
            let m1 = CsrMatrix::from(m.clone());
            assert!(m1.iter().eq(m.entries()));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_add_1() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_add_pair::<i32, _, _>(DokMatrix::arb_fixed_size_matrix),
            |AddPair(m1, m2)| {
                assert_eq!(
                    CsrMatrix::from(m1.clone()) + CsrMatrix::from(m2.clone()),
                    CsrMatrix::from(m1 + m2)
                );
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn test_add_2() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_add_pair::<i32, _, _>(CsrMatrix::arb_fixed_size_matrix),
            |AddPair(m1, m2)| {
                let m = m1.clone() + m2.clone();
                assert!(iproduct!(0..m.rows(), 0..m.cols()).all(|p| {
                    m1.get_element(p).into_owned() + m2.get_element(p).into_owned()
                        == m.get_element(p).into_owned()
                }));
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn test_mul_1() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<i32, _, _>(DokMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| {
                assert_eq!(
                    &CsrMatrix::from(m1.clone()) * &CsrMatrix::from(m2.clone()),
                    CsrMatrix::from(&m1 * &m2)
                );
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn test_mul_2() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<i32, _, _>(CsrMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| {
                let m = &m1.clone() * &m2.clone();
                assert!(iproduct!(0..m.rows(), 0..m.cols()).all(|(i, j)| {
                    m.get_element((i, j)).into_owned()
                        == (0..m1.cols())
                            .map(|k| {
                                m1.get_element((i, k)).into_owned()
                                    * m2.get_element((k, j)).into_owned()
                            })
                            .sum()
                }));
                Ok(())
            },
        )
        .unwrap();
}
