use arbitrary::{Arbitrary, Unstructured};
use num::Num;

use crate::{dok_matrix::DokMatrix, AddPair, Matrix, MatrixError, MulPair};

pub fn arb_fixed_size_matrix<'a, T: arbitrary::Arbitrary<'a> + Num + Clone, M: Matrix<T>>(
    u: &mut arbitrary::Unstructured<'a>,
    rows: usize,
    cols: usize,
) -> arbitrary::Result<Result<M, MatrixError>> {
    if rows == 0 || cols == 0 {
        return Ok(Err(MatrixError::HasZeroDimension));
    }
    let mut matrix = M::new(rows, cols).unwrap();
    for _ in 0..u.arbitrary_len::<(usize, usize, T)>()? {
        let i = u.int_in_range(0..=rows - 1)?;
        let j = u.int_in_range(0..=cols - 1)?;
        matrix.set_element((i, j), u.arbitrary()?);
    }
    Ok(Ok(matrix))
}

pub fn arb_add_pair_fixed_size<'a, T: Arbitrary<'a> + Num + Clone, M: Matrix<T>>(
    u: &mut Unstructured<'a>,
    rows: usize,
    cols: usize,
) -> arbitrary::Result<Result<AddPair<M>, MatrixError>> {
    let res1 = arb_fixed_size_matrix(u, rows, cols)?;
    let res2 = arb_fixed_size_matrix(u, rows, cols)?;
    Ok(res1.and_then(|m1| res2.map(|m2| AddPair(m1, m2))))
}

pub fn arb_mul_pair_fixed_size<'a, T: Arbitrary<'a> + Num + Clone, M: Matrix<T>>(
    u: &mut Unstructured<'a>,
    l: usize,
    m: usize,
    n: usize,
) -> arbitrary::Result<Result<MulPair<DokMatrix<T>>, MatrixError>> {
    let res1 = arb_fixed_size_matrix(u, l, m)?;
    let res2 = arb_fixed_size_matrix(u, m, n)?;
    Ok(res1.and_then(|m1| res2.map(|m2| MulPair(m1, m2))))
}
