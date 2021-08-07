use std::{convert::TryInto, num::NonZeroUsize};

use num::{Complex, Num};
use proptest::{
    arbitrary::{any, Arbitrary},
    strategy::Strategy,
};

use crate::{dok_matrix::DokMatrix, AddPair, ComplexNewtype, Matrix, MulPair};

const MAX_SIZE: usize = 20;

pub fn arb_matrix<T: Arbitrary, F: Fn(NonZeroUsize, NonZeroUsize) -> S, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = S::Value> {
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(move |(rows, cols)| {
        arb_matrix_fixed_size(rows.try_into().unwrap(), cols.try_into().unwrap())
    })
}

pub fn arb_add_pair_fixed_size<
    T: Arbitrary + Clone + Num,
    F: Fn(NonZeroUsize, NonZeroUsize) -> S,
    S: Strategy,
>(
    rows: NonZeroUsize,
    cols: NonZeroUsize,
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = AddPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    arb_matrix_fixed_size(rows, cols).prop_flat_map(move |m| {
        arb_matrix_fixed_size(m.rows(), m.cols()).prop_map(move |m1| AddPair(m.clone(), m1))
    })
}

pub fn arb_add_pair<
    T: Arbitrary + Clone + Num,
    F: Fn(NonZeroUsize, NonZeroUsize) -> S + Copy,
    S: Strategy,
>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = AddPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(move |(rows, cols)| {
        arb_add_pair_fixed_size(
            rows.try_into().unwrap(),
            cols.try_into().unwrap(),
            arb_matrix_fixed_size,
        )
    })
}

pub fn arb_mul_pair_fixed_size<
    T: Arbitrary + Clone + Num,
    F: Fn(NonZeroUsize, NonZeroUsize) -> S,
    S: Strategy,
>(
    l: usize,
    n: usize,
    p: usize,
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = MulPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    let n = n.try_into().unwrap();
    arb_matrix_fixed_size(l.try_into().unwrap(), n).prop_flat_map(move |m| {
        arb_matrix_fixed_size(n, p.try_into().unwrap()).prop_map(move |m1| MulPair(m.clone(), m1))
    })
}

pub fn arb_mul_pair<
    T: Arbitrary + Clone + Num,
    F: Fn(NonZeroUsize, NonZeroUsize) -> S + Copy,
    S: Strategy,
>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = MulPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    (1..MAX_SIZE, 1..MAX_SIZE, 1..MAX_SIZE)
        .prop_flat_map(move |(l, n, p)| arb_mul_pair_fixed_size(l, n, p, arb_matrix_fixed_size))
}

impl<T: Arbitrary + Num + Clone> DokMatrix<T> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: NonZeroUsize,
        cols: NonZeroUsize,
    ) -> impl Strategy<Value = Self> {
        proptest::collection::vec(
            ((0..rows.get(), 0..cols.get()), T::arbitrary()),
            0..=(2 * rows.get() * cols.get()),
        )
        .prop_map(move |entries| {
            let mut m = DokMatrix::new((rows, cols));
            for (pos, t) in entries {
                m.set_element(pos, t).unwrap();
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
