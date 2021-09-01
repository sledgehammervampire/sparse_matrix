use std::num::NonZeroUsize;

use arbitrary::{Arbitrary, Unstructured};

use crate::{AddPair, Matrix, MulPair};

pub fn arb_fixed_size_matrix<'a, T: Arbitrary<'a>, M: Matrix<T>>(
    u: &mut arbitrary::Unstructured<'a>,
    rows: NonZeroUsize,
    cols: NonZeroUsize,
) -> arbitrary::Result<M> {
    let mut matrix = M::new((rows, cols));
    for _ in
        0..u.int_in_range(0..=1_000.min(rows.get().saturating_mul(cols.get()).saturating_add(5)))?
    {
        let i = u.int_in_range(0..=rows.get() - 1)?;
        let j = u.int_in_range(0..=cols.get() - 1)?;
        matrix.set_element((i, j), u.arbitrary()?).unwrap();
    }
    Ok(matrix)
}

pub fn arb_add_pair_fixed_size<'a, T: Arbitrary<'a>, M: Matrix<T>>(
    u: &mut Unstructured<'a>,
    rows: NonZeroUsize,
    cols: NonZeroUsize,
) -> arbitrary::Result<AddPair<M>> {
    let m1 = arb_fixed_size_matrix(u, rows, cols)?;
    let m2 = arb_fixed_size_matrix(u, rows, cols)?;
    Ok(AddPair(m1, m2))
}

pub fn arb_mul_pair_fixed_size<'a, T: Arbitrary<'a>, M: Matrix<T>>(
    u: &mut Unstructured<'a>,
    l: NonZeroUsize,
    m: NonZeroUsize,
    n: NonZeroUsize,
) -> arbitrary::Result<MulPair<M>> {
    let m1 = arb_fixed_size_matrix(u, l, m)?;
    let m2 = arb_fixed_size_matrix(u, m, n)?;
    Ok(MulPair(m1, m2))
}
