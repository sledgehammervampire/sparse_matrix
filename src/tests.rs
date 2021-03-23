use itertools::iproduct;
use num::Num;
use proptest::{
    arbitrary::any,
    strategy::{Just, Strategy},
    test_runner::TestRunner,
};

use crate::{
    arbitrary::{
        arb_add_pair, arb_csr_matrix, arb_csr_matrix_fixed_size, arb_mul_pair, AddPair, MulPair,
    },
    dok_matrix::{arb_dok_matrix, arb_dok_matrix_fixed_size, DokMatrix},
    CsrMatrix, Matrix,
};

fn csr_invariants<T: Num>(m: &CsrMatrix<T>) -> bool {
    csr_invariant_1(m)
        && csr_invariant_2(m)
        && csr_invariant_3(m)
        && csr_invariant_4(m)
        && csr_invariant_5(m)
        && csr_invariant_6(m)
        && csr_invariant_7(m)
        && csr_invariant_8(m)
}

fn csr_invariant_1<T>(m: &CsrMatrix<T>) -> bool {
    m.ridx
        .iter()
        .all(|(i, s1)| m.ridx.range(..i).fold(0, |sum, (_, s2)| sum + s2.len) == s1.start)
}

fn csr_invariant_2<T>(m: &CsrMatrix<T>) -> bool {
    m.ridx.values().map(|s| s.len).sum::<usize>() == m.vals.len()
}

fn csr_invariant_3<T>(m: &CsrMatrix<T>) -> bool {
    m.cidx.len() == m.vals.len()
}

fn csr_invariant_4<T>(m: &CsrMatrix<T>) -> bool {
    m.ridx.values().all(|s| s.len > 0)
}

fn csr_invariant_5<T>(m: &CsrMatrix<T>) -> bool {
    fn is_increasing(s: &[usize]) -> bool {
        let mut max = None;
        for i in s {
            if Some(i) > max {
                max = Some(i);
            } else {
                return false;
            }
        }
        true
    }

    m.ridx
        .values()
        .all(|s| is_increasing(&m.cidx[s.start..s.start + s.len]))
}

fn csr_invariant_6<T: Num>(m: &CsrMatrix<T>) -> bool {
    m.vals.iter().all(|t| !t.is_zero())
}

fn csr_invariant_7<T: Num>(m: &CsrMatrix<T>) -> bool {
    m.ridx.keys().all(|r| (0..m.rows).contains(r))
}

fn csr_invariant_8<T: Num>(m: &CsrMatrix<T>) -> bool {
    m.cidx.iter().all(|c| (0..m.cols).contains(c))
}

#[test]
fn test_csr_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&arb_csr_matrix::<i32>(), |m| {
            assert!(csr_invariants(&m));
            Ok(())
        })
        .unwrap();
}

#[test]
fn test_dok_invariants() {
    let mut runner = TestRunner::default();
    runner
        .run(&arb_dok_matrix::<i32>(), |m| {
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
            &arb_csr_matrix::<i32>()
                .prop_flat_map(|m| (0..m.rows, 0..m.cols, Just(m), any::<i32>())),
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
            &arb_dok_matrix::<i32>().prop_flat_map(|m| {
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
