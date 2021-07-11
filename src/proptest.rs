use num::{Complex, Num};
use proptest::{
    arbitrary::{any, Arbitrary},
    strategy::Strategy,
};

use crate::{dok_matrix::DokMatrix, AddPair, ComplexNewtype, Matrix, MulPair};

const MAX_SIZE: usize = 20;

pub fn arb_matrix<T: Arbitrary, F: Fn(usize, usize) -> S, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = S::Value> {
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(move |(rows, cols)| arb_matrix_fixed_size(rows, cols))
}

pub fn arb_add_pair_fixed_size<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S, S: Strategy>(
    rows: usize,
    cols: usize,
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = AddPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    arb_matrix_fixed_size(rows, cols).prop_flat_map(move |m| {
        arb_matrix_fixed_size(m.rows(), m.cols()).prop_map(move |m1| AddPair(m.clone(), m1))
    })
}

pub fn arb_add_pair<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S + Copy, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = AddPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(move |(rows, cols)| {
        arb_add_pair_fixed_size(rows, cols, arb_matrix_fixed_size)
    })
}

pub fn arb_mul_pair_fixed_size<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S, S: Strategy>(
    l: usize,
    n: usize,
    p: usize,
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = MulPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    arb_matrix_fixed_size(l, n).prop_flat_map(move |m| {
        arb_matrix_fixed_size(n, p).prop_map(move |m1| MulPair(m.clone(), m1))
    })
}

pub fn arb_mul_pair<T: Arbitrary + Clone + Num, F: Fn(usize, usize) -> S + Copy, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = MulPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    (1..MAX_SIZE, 1..MAX_SIZE, 1..MAX_SIZE)
        .prop_flat_map(move |(l, n, p)| arb_mul_pair_fixed_size(l, n, p, arb_matrix_fixed_size))
}

impl<T: Arbitrary + Num + Clone> DokMatrix<T> {
    pub(crate) fn arb_fixed_size_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Self> {
        proptest::collection::vec(((0..rows, 0..cols), T::arbitrary()), 0..=(2 * rows * cols))
            .prop_map(move |entries| {
                let mut m = DokMatrix::new(rows, cols).unwrap();
                for (pos, t) in entries {
                    m.set_element(pos, t);
                }
                m
            })
    }

    pub fn arb_matrix() -> impl Strategy<Value = Self> {
        arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

impl<T: Arbitrary> Arbitrary for ComplexNewtype<T> {
    type Parameters = ();
    type Strategy =
        proptest::strategy::Map<(T::Strategy, T::Strategy), fn((T, T)) -> ComplexNewtype<T>>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (any::<T>(), any::<T>()).prop_map(|(re, im)| ComplexNewtype(Complex { re, im }))
    }
}
