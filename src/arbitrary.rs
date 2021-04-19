use arbitrary::{Arbitrary, Result, Unstructured};
use num::Num;

use crate::{csr_matrix::CsrMatrix, dok_matrix::DokMatrix, AddPair, Matrix, MulPair};

pub fn arb_fixed_size_matrix<'a, T: arbitrary::Arbitrary<'a> + Num + Clone, M: Matrix<T>>(
    u: &mut arbitrary::Unstructured<'a>,
    rows: usize,
    cols: usize,
) -> arbitrary::Result<M> {
    let nnz = u.int_in_range(0..=rows.saturating_mul(cols))?;
    let mut m = M::new(rows, cols);
    for _ in 0..nnz {
        m.set_element(
            (u.int_in_range(0..=rows - 1)?, u.int_in_range(0..=cols - 1)?),
            u.arbitrary()?,
        );
    }
    Ok(m)
}

impl<'a, T: arbitrary::Arbitrary<'a> + Num + Clone> arbitrary::Arbitrary<'a> for DokMatrix<T> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let rows = u.arbitrary_len::<T>()?;
        let cols = u.arbitrary_len::<T>()?;
        arb_fixed_size_matrix(u, rows, cols)
    }
}

impl<'a, T: arbitrary::Arbitrary<'a> + Num + Clone> arbitrary::Arbitrary<'a> for CsrMatrix<T> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(CsrMatrix::from(u.arbitrary::<DokMatrix<T>>()?))
    }
}

fn arb_add_pair_fixed_size<'a, T: Arbitrary<'a> + Num + Clone>(
    u: &mut Unstructured<'a>,
    rows: usize,
    cols: usize,
) -> arbitrary::Result<AddPair<DokMatrix<T>>> {
    Ok(AddPair(
        arb_fixed_size_matrix(u, rows, cols)?,
        arb_fixed_size_matrix(u, rows, cols)?,
    ))
}

impl<'a, T: Arbitrary<'a> + Num + Clone> Arbitrary<'a> for AddPair<DokMatrix<T>> {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let rows = u.arbitrary_len::<T>()?;
        let cols = u.arbitrary_len::<T>()?;
        arb_add_pair_fixed_size(u, rows, cols)
    }
}

impl<'a, T: Arbitrary<'a> + Num + Clone> Arbitrary<'a> for AddPair<CsrMatrix<T>> {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let AddPair(m1, m2) = u.arbitrary::<AddPair<DokMatrix<T>>>()?;
        Ok(AddPair(CsrMatrix::from(m1), CsrMatrix::from(m2)))
    }
}

fn arb_mul_pair_fixed_size<'a, T: Arbitrary<'a> + Num + Clone>(
    u: &mut Unstructured<'a>,
    l: usize,
    m: usize,
    n: usize,
) -> Result<MulPair<DokMatrix<T>>> {
    Ok(MulPair(
        arb_fixed_size_matrix(u, l, m)?,
        arb_fixed_size_matrix(u, m, n)?,
    ))
}

impl<'a, T: Arbitrary<'a> + Num + Clone> Arbitrary<'a> for MulPair<DokMatrix<T>> {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let l = u.arbitrary_len::<T>()?;
        let m = u.arbitrary_len::<T>()?;
        let n = u.arbitrary_len::<T>()?;
        arb_mul_pair_fixed_size(u, l, m, n)
    }
}

impl<'a, T: Arbitrary<'a> + Num + Clone> Arbitrary<'a> for MulPair<CsrMatrix<T>> {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let MulPair(m1, m2) = u.arbitrary::<MulPair<DokMatrix<T>>>()?;
        Ok(MulPair(CsrMatrix::from(m1), CsrMatrix::from(m2)))
    }
}
