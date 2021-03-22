use itertools::iproduct;
use proptest::test_runner::TestRunner;

use crate::{
    dok_matrix::{
        arb_add_pair, arb_csr_matrix, arb_csr_matrix_fixed_size, arb_dok_matrix,
        arb_dok_matrix_fixed_size, arb_mul_pair, AddPair, DokMatrix, MulPair,
    },
    CsrMatrix, Matrix,
};

#[test]
fn test_convert_1() {
    let mut runner = TestRunner::default();
    runner
        .run(&arb_dok_matrix::<i32>(), |m| {
            assert_eq!(m, DokMatrix::from(CsrMatrix::from(m.clone())));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_convert_2() {
    let mut runner = TestRunner::default();
    runner
        .run(&arb_dok_matrix::<i32>(), |m| {
            let m1 = CsrMatrix::from(m.clone());
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
        .run(&arb_csr_matrix::<i32>(), |m| {
            assert_eq!(m, CsrMatrix::from(DokMatrix::from(m.clone())));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_convert_4() {
    let mut runner = TestRunner::default();
    runner
        .run(&arb_csr_matrix::<i32>(), |m| {
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
        .run(&arb_csr_matrix::<i32>(), |m| {
            let m1 = m.transpose();
            assert!(iproduct!(0..m.rows, 0..m.cols)
                .all(|(i, j)| m.get_element((i, j)) == m1.get_element((j, i))));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_iter() {
    let mut runner = TestRunner::default();
    runner
        .run(&arb_dok_matrix::<i32>(), |m| {
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
            &arb_add_pair::<i32, _, _>(arb_dok_matrix_fixed_size),
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
            &arb_add_pair::<i32, _, _>(arb_csr_matrix_fixed_size),
            |AddPair(m1, m2)| {
                let m = m1.clone() + m2.clone();
                assert!(iproduct!(0..m.rows, 0..m.cols).all(|p| {
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
            &arb_mul_pair::<i32, _, _>(arb_dok_matrix_fixed_size),
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
            &arb_mul_pair::<i32, _, _>(arb_csr_matrix_fixed_size),
            |MulPair(m1, m2)| {
                let m = &m1.clone() * &m2.clone();
                assert!(iproduct!(0..m.rows, 0..m.cols).all(|(i, j)| {
                    m.get_element((i, j)).into_owned()
                        == (0..m1.cols)
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
