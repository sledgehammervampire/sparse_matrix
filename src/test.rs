use itertools::iproduct;
use proptest::prelude::*;

use crate::{
    dok_matrix::{arb_add_pair, arb_matrix, AddPair, DokMatrix},
    CsrMatrix,
};

proptest! {
    #[test]
    fn test_convert(m in arb_matrix::<i32>()) {
        assert_eq!(m, DokMatrix::from(CsrMatrix::from(m.clone())));
    }

    #[test]
    fn test_transpose(m in arb_matrix::<i32>()) {
        let m = CsrMatrix::from(m);
        let m1 = m.transpose();
        assert!(iproduct!(0..m.rows, 0..m.cols)
            .all(|(i, j)| m.get_element((i, j)) == m1.get_element((j, i))));
    }

    #[test]
    fn test_iter(m in arb_matrix::<i32>()) {
        let m1 = CsrMatrix::from(m.clone());
        assert!(m1.iter().eq(m.entries()));
    }

    #[test]
    fn test_add_1(AddPair(mut m1, m2) in arb_add_pair::<i32>()) {
        let mut m3 = CsrMatrix::from(m1.clone());
        let m4 = CsrMatrix::from(m2.clone());
        m3 += &m4;
        m1 += &m2;
        assert_eq!(CsrMatrix::from(m1), m3);
    }

    #[test]
    fn test_add_2(AddPair(m1, m2) in arb_add_pair::<i32>()) {
        let (m1, m2) = (CsrMatrix::from(m1), CsrMatrix::from(m2));
        let mut m = m1.clone();
        m += &m2;
        assert!(iproduct!(0..m.rows, 0..m.cols)
            .all(|p| {
                m.get_element(p).into_owned()
                    == m1.get_element(p).into_owned() + m2.get_element(p).as_ref()
            }));
    }
}

/*
#[test]
fn test_mul() {
    fn prop_mul_1(MulPair(mut m1, m2): MulPair<i32>) -> bool {
        let mut m3 = CsrMatrix::from(m1.clone());
        let m4 = CsrMatrix::from(m2.clone());
        m3 *= &m4;
        m1 *= &m2;
        CsrMatrix::from(m1) == m3
    }
    fn prop_mul_2(MulPair(m1, m2): MulPair<i32>) -> bool {
        let (m1, m2) = (CsrMatrix::from(m1), CsrMatrix::from(m2));
        let mut m = m1.clone();
        m *= &m2;
        (0..m.rows)
            .flat_map(|i| (0..m.cols).map(move |j| (i, j)))
            .all(|(i, j)| {
                m.get_element((i, j)).into_owned()
                    == (0..m1.cols)
                        .map(|k| {
                            m1.get_element((i, k)).into_owned() * m2.get_element((k, j)).as_ref()
                        })
                        .sum()
            })
    }
    QuickCheck::new()
        .tests(1 << 10)
        .quickcheck(prop_mul_1 as fn(_) -> bool);
    QuickCheck::new()
        .tests(1 << 10)
        .quickcheck(prop_mul_2 as fn(_) -> bool);
}
*/
