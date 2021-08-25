use std::{convert::TryInto, num::NonZeroUsize};

use proptest::prelude::*;

use crate::{AddPair, Matrix, MulPair};

const MAX_SIZE: usize = 5;

pub fn arb_matrix<T, F: Fn(NonZeroUsize, NonZeroUsize) -> S, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = S::Value>
where
    S::Value: Matrix<T>,
{
    (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(move |(rows, cols)| {
        arb_matrix_fixed_size(rows.try_into().unwrap(), cols.try_into().unwrap())
    })
}

pub fn arb_add_pair_fixed_size<T, F: Fn(NonZeroUsize, NonZeroUsize) -> S, S: Strategy>(
    rows: NonZeroUsize,
    cols: NonZeroUsize,
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = AddPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    arb_matrix_fixed_size(rows, cols).prop_flat_map(move |m| {
        arb_matrix_fixed_size(rows, cols).prop_map(move |m1| AddPair(m.clone(), m1))
    })
}

pub fn arb_add_pair<T, F: Fn(NonZeroUsize, NonZeroUsize) -> S + Copy, S: Strategy>(
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

pub fn arb_mul_pair_fixed_size<T, F: Fn(NonZeroUsize, NonZeroUsize) -> S, S: Strategy>(
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

pub fn arb_mul_pair<T, F: Fn(NonZeroUsize, NonZeroUsize) -> S + Copy, S: Strategy>(
    arb_matrix_fixed_size: F,
) -> impl Strategy<Value = MulPair<S::Value>>
where
    S::Value: Matrix<T> + Clone,
{
    (1..MAX_SIZE, 1..MAX_SIZE, 1..MAX_SIZE)
        .prop_flat_map(move |(l, n, p)| arb_mul_pair_fixed_size(l, n, p, arb_matrix_fixed_size))
}
