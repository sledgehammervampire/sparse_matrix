#![allow(clippy::disallowed_method)]

use cap_rand::{ambient_authority, prelude::*};
use itertools::iproduct;
use num_traits::Num;
use open_ambient::open_ambient_dir;
use proptest::{prelude::*, test_runner::TestRunner};
use spam_dok::{parse_matrix_market, DokMatrix, MatrixType};
use spam_matrix::{
    proptest::{arb_add_pair, arb_matrix, arb_mul_pair},
    AddPair, Matrix, MulPair,
};
use std::{
    convert::TryInto,
    fmt::Debug,
    io::Read,
    iter::repeat_with,
    num::{NonZeroUsize, Wrapping},
};

use crate::CsrMatrix;

const MAX_SIZE: usize = 10;

impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T, false> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: NonZeroUsize,
        cols: NonZeroUsize,
    ) -> impl proptest::strategy::Strategy<Value = Self> {
        use proptest::prelude::*;
        repeat_with(|| proptest::collection::hash_set(0..cols.get(), 0..=cols.get()))
            .take(rows.get())
            .collect::<Vec<_>>()
            .prop_flat_map(move |cidx| {
                let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
                for rcidx in cidx {
                    ridx.push(ridx.last().unwrap() + rcidx.len());
                    cidx_flattened.extend(rcidx);
                }
                repeat_with(T::arbitrary)
                    .take(cidx_flattened.len())
                    .collect::<Vec<_>>()
                    .prop_map(move |vals| CsrMatrix {
                        rows,
                        cols,
                        vals,
                        indices: cidx_flattened.clone(),
                        offsets: ridx.clone(),
                    })
            })
    }

    fn arb_matrix() -> impl proptest::strategy::Strategy<Value = Self> {
        arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T, true> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: NonZeroUsize,
        cols: NonZeroUsize,
    ) -> impl proptest::strategy::Strategy<Value = Self> {
        use proptest::prelude::*;
        repeat_with(|| {
            proptest::sample::subsequence((0..cols.get()).collect::<Vec<_>>(), 0..=cols.get())
        })
        .take(rows.get())
        .collect::<Vec<_>>()
        .prop_flat_map(move |cidx| {
            let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
            for mut rcidx in cidx {
                ridx.push(ridx.last().unwrap() + rcidx.len());
                cidx_flattened.append(&mut rcidx);
            }
            repeat_with(T::arbitrary)
                .take(cidx_flattened.len())
                .collect::<Vec<_>>()
                .prop_map(move |vals| CsrMatrix {
                    rows,
                    cols,
                    vals,
                    indices: cidx_flattened.clone(),
                    offsets: ridx.clone(),
                })
        })
    }

    pub fn arb_matrix() -> impl proptest::strategy::Strategy<Value = Self> {
        arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

/*
   the following diagram commutes:

                     dok_op
                  . -------> .
                  |          ^
CsrMatrix::from   |          | DokMatrix::from
                  v          |
                  . -------> .
                     csr_op
    */
fn test_commutes<T, U, V, S, F, G, H>(strategy: &S, dok_val: F, gen_csr: G, into_val: H)
where
    T: Num + Debug,
    U: Matrix<T>,
    V: Eq + Debug,
    S: Strategy,
    S::Value: Clone,
    F: Fn(S::Value) -> V,
    G: Fn(S::Value) -> U,
    H: Fn(U) -> V,
{
    let mut runner = TestRunner::default();
    runner
        .run(strategy, |v| {
            let m = gen_csr(v.clone());
            prop_assert!(m.invariants());
            prop_assert_eq!(into_val(m), dok_val(v));
            Ok(())
        })
        .unwrap();
}

// base cases
#[test]
fn new() {
    let strategy = (1..MAX_SIZE, 1..MAX_SIZE)
        .prop_map(|(m, n)| (m.try_into().unwrap(), n.try_into().unwrap()));
    test_commutes(
        &strategy,
        DokMatrix::<i8>::new,
        CsrMatrix::<_, false>::new,
        DokMatrix::from,
    );
    test_commutes(
        &strategy,
        DokMatrix::<i8>::new,
        CsrMatrix::<_, true>::new,
        DokMatrix::from,
    );
}

#[test]
fn identity() {
    let strategy = (1..MAX_SIZE).prop_map(|m| m.try_into().unwrap());
    test_commutes(
        &strategy,
        DokMatrix::<i8>::identity,
        CsrMatrix::<_, false>::identity,
        DokMatrix::from,
    );
    test_commutes(
        &strategy,
        DokMatrix::<i8>::identity,
        CsrMatrix::<_, true>::identity,
        DokMatrix::from,
    );
}

#[test]
fn arb() {
    let mut runner = TestRunner::default();
    runner
        .run(&CsrMatrix::<i8, false>::arb_matrix(), |m| {
            prop_assert!(m.invariants(), "{:?}", m);
            Ok(())
        })
        .unwrap();
    runner
        .run(&CsrMatrix::<i8, true>::arb_matrix(), |m| {
            prop_assert!(m.invariants(), "{:?}", m);
            Ok(())
        })
        .unwrap();
}

#[test]
fn from_arb_dok() {
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        |m| m,
        |m| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);
            CsrMatrix::from_dok(m, &mut rng)
        },
        DokMatrix::from,
    );
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        |m| m,
        CsrMatrix::from,
        DokMatrix::from,
    );
}

// #[test]
// fn from_arb_csr() {
//     test_invariants(&CsrMatrix::<i8, false>::arb_matrix(), DokMatrix::from);
//     test_invariants(&CsrMatrix::<i8, true>::arb_matrix(), DokMatrix::from);
// }

// other
#[test]
fn rows() {
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        |m| DokMatrix::rows(&m),
        |m| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);
            CsrMatrix::from_dok(m, &mut rng)
        },
        |m| CsrMatrix::rows(&m),
    );
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        |m| DokMatrix::rows(&m),
        CsrMatrix::from,
        |m| CsrMatrix::rows(&m),
    );
}

#[test]
fn cols() {
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        |m| DokMatrix::cols(&m),
        |m| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);
            CsrMatrix::from_dok(m, &mut rng)
        },
        |m| CsrMatrix::cols(&m),
    );
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        |m| DokMatrix::cols(&m),
        CsrMatrix::from,
        |m| CsrMatrix::cols(&m),
    );
}

#[test]
fn get_element() {
    let mut runner = TestRunner::default();
    runner
        .run(&DokMatrix::<i8>::arb_matrix(), |m| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);

            let m1 = CsrMatrix::from_dok(m.clone(), &mut rng);
            prop_assert!(
                iproduct!(0..m.rows().get(), 0..m.cols().get())
                    .all(|pos| m.get_element(pos) == m1.get_element(pos)),
                "{:?}",
                m1
            );
            let m1 = CsrMatrix::from(m.clone());
            prop_assert!(
                iproduct!(0..m.rows().get(), 0..m.cols().get())
                    .all(|pos| m.get_element(pos) == m1.get_element(pos)),
                "{:?}",
                m1
            );
            Ok(())
        })
        .unwrap();
}

// inductive cases
#[test]
fn set_element() {
    let strategy = (1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(|(rows, cols)| {
        (
            DokMatrix::<i8>::arb_fixed_size_matrix(
                rows.try_into().unwrap(),
                cols.try_into().unwrap(),
            ),
            0..rows,
            0..cols,
            any::<i8>(),
        )
    });
    let mut runner = TestRunner::default();
    runner
        .run(&strategy, |(mut m, i, j, t)| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);
            let mut m1 = CsrMatrix::from_dok(m.clone(), &mut rng);
            let t1_old = m1.set_element((i, j), t).unwrap();
            assert!(m1.invariants());
            let t_old = m.set_element((i, j), t).unwrap();
            assert_eq!(m, DokMatrix::from(m1));
            assert_eq!(t_old, t1_old);
            Ok(())
        })
        .unwrap();
    runner
        .run(&strategy, |(mut m, i, j, t)| {
            let mut m1 = CsrMatrix::from(m.clone());
            let t1_old = m1.set_element((i, j), t).unwrap();
            assert!(m1.invariants());
            let t_old = m.set_element((i, j), t).unwrap();
            assert_eq!(m, DokMatrix::from(m1));
            assert_eq!(t_old, t1_old);
            Ok(())
        })
        .unwrap();
}

#[test]
fn transpose() {
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        DokMatrix::transpose,
        |m| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);
            CsrMatrix::from_dok(m, &mut rng).transpose()
        },
        DokMatrix::from,
    );
    test_commutes(
        &DokMatrix::<i8>::arb_matrix(),
        DokMatrix::transpose,
        |m| CsrMatrix::from(m).transpose(),
        DokMatrix::from,
    );
}

#[test]
fn add() {
    let strategy = arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix)
        .prop_map(|AddPair(m1, m2)| (m1, m2));
    test_commutes(
        &strategy,
        |(m1, m2)| m1 + m2,
        |(m1, m2)| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);
            CsrMatrix::from_dok(m1, &mut rng) + CsrMatrix::from_dok(m2, &mut rng)
        },
        DokMatrix::from,
    );
    test_commutes(
        &strategy,
        |(m1, m2)| m1 + m2,
        |(m1, m2)| CsrMatrix::from(m1) + CsrMatrix::from(m2),
        DokMatrix::from,
    );
}

#[test]
fn mul_hash() {
    let strategy = arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix)
        .prop_map(|MulPair(m1, m2)| (m1, m2));
    test_commutes(
        &strategy,
        |(m1, m2)| &m1 * &m2,
        |(m1, m2)| {
            let ambient_authority = ambient_authority();
            let mut rng = CapRng::default(ambient_authority);
            CsrMatrix::from_dok(m1, &mut rng)
                .mul_hash::<false, false>(&CsrMatrix::from_dok(m2, &mut rng))
        },
        DokMatrix::from,
    );
}

#[ignore = "expensive, parsing code not changed often"]
#[test]
fn from_matrix_market() {
    fn inner<T: Num + Clone>(m: DokMatrix<T>, rng: &mut CapRng) {
        let m1 = CsrMatrix::from_dok(m.clone(), rng);
        assert!(m1.invariants());
        let m1 = CsrMatrix::from(m);
        assert!(m1.invariants());
    }
    let ambient_authority = ambient_authority();
    let mut rng = CapRng::default(ambient_authority);
    let dir = open_ambient_dir!("../matrices").unwrap();
    for entry in dir.entries().unwrap() {
        let mut ser = String::new();
        entry
            .unwrap()
            .open()
            .unwrap()
            .read_to_string(&mut ser)
            .unwrap();
        match parse_matrix_market::<i64, f64>(&ser).unwrap() {
            MatrixType::Integer(m) => {
                inner(m, &mut rng);
            }
            MatrixType::Real(m) => {
                inner(m, &mut rng);
            }
            MatrixType::Complex(m) => {
                inner(m, &mut rng);
            }
        };
    }
}

#[test]
fn test_into_iter() {
    use proptest::prop_assert_eq;

    let config = proptest::test_runner::Config {
        max_shrink_iters: 10000,
        ..Default::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(&CsrMatrix::<i8, false>::arb_matrix(), |m| {
            prop_assert_eq!(
                m.iter().map(|e| (e.0, *e.1)).collect::<Vec<_>>(),
                m.into_iter().collect::<Vec<_>>()
            );
            Ok(())
        })
        .unwrap();
}
