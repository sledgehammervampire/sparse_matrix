use itertools::iproduct;
use proptest::prelude::*;

use crate::{
    dok_matrix::{arb_add_pair, arb_matrix, arb_mul_pair, AddPair, DokMatrix, MulPair},
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
    fn test_add_1(AddPair(m1, m2) in arb_add_pair::<i32>()) {
        assert_eq!(
            CsrMatrix::from(m1.clone()) + CsrMatrix::from(m2.clone()),
            CsrMatrix::from(m1 + m2)
        );
    }

    #[test]
    fn test_add_2(AddPair(m1, m2) in arb_add_pair::<i32>()) {
        let (m1, m2) = (CsrMatrix::from(m1), CsrMatrix::from(m2));
        let m = m1.clone() + m2.clone();
        assert!(iproduct!(0..m.rows, 0..m.cols).all(|p| {
            m1.get_element(p).into_owned() + m2.get_element(p).into_owned()
                == m.get_element(p).into_owned()
        }));
    }

    #[test]
    fn test_mul_1(MulPair(m1, m2) in arb_mul_pair::<i32>()) {
        assert_eq!(
            &CsrMatrix::from(m1.clone()) * &CsrMatrix::from(m2.clone()),
            CsrMatrix::from(&m1 * &m2)
        );
    }

    #[test]
    fn test_mul_2(MulPair(m1, m2) in arb_mul_pair::<i32>()) {
        let (m1, m2) = (CsrMatrix::from(m1), CsrMatrix::from(m2));
        let m = &m1.clone() * &m2.clone();
        assert!(iproduct!(0..m.rows, 0..m.cols).all(|(i, j)| {
            m.get_element((i, j)).into_owned()
                == (0..m1.cols)
                    .map(|k| m1.get_element((i, k)).into_owned() * m2.get_element((k, j)).into_owned())
                    .sum()
        }));
    }
}
