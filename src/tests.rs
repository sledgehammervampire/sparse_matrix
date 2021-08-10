#![allow(clippy::disallowed_method)]
mod dok {
    use std::{convert::TryInto, fs, num::Wrapping};

    use cap_rand::{ambient_authority, prelude::CapRng};
    use num::Num;
    use proptest::{arbitrary::any, prop_assert, strategy::Strategy, test_runner::TestRunner};
    use walkdir::WalkDir;

    use crate::{
        csr_matrix::CsrMatrix,
        dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
        proptest::{arb_add_pair, arb_mul_pair},
        AddPair, Matrix, MulPair,
    };

    const MAX_SIZE: usize = 10;

    fn test_invariants<S, F, T>(strategy: &S, f: F)
    where
        S: Strategy,
        T: Num,
        F: Fn(S::Value) -> DokMatrix<T>,
    {
        let mut runner = TestRunner::default();
        runner
            .run(strategy, |v| {
                let m = f(v);
                prop_assert!(m.invariants());
                Ok(())
            })
            .unwrap();
    }

    // base cases
    #[test]
    fn new_invariants() {
        test_invariants(
            &(1..MAX_SIZE, 1..MAX_SIZE)
                .prop_map(|(m, n)| (m.try_into().unwrap(), n.try_into().unwrap())),
            DokMatrix::<i8>::new,
        );
    }

    #[test]
    fn identity_invariants() {
        test_invariants(
            &(1..MAX_SIZE).prop_map(|n| n.try_into().unwrap()),
            DokMatrix::<i8>::identity,
        );
    }

    #[test]
    fn arb_invariants() {
        test_invariants(&DokMatrix::<i8>::arb_matrix(), |m| m);
    }

    #[test]
    fn from_arb_csr_invariants() {
        test_invariants(&CsrMatrix::<i8, false>::arb_matrix(), DokMatrix::from);
        test_invariants(&CsrMatrix::<i8, true>::arb_matrix(), DokMatrix::from);
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
        for entry in WalkDir::new("matrices") {
            let entry = entry.unwrap();
            if let Some(ext) = entry.path().extension() {
                if ext == "mtx" {
                    match parse_matrix_market::<i64, f64>(
                        &fs::read_to_string(entry.path()).unwrap(),
                    )
                    .unwrap()
                    {
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
        }
    }

    // inductive cases
    #[test]
    fn add() {
        test_invariants(
            &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |AddPair(m1, m2)| m1 + m2,
        );
    }

    #[test]
    fn mul() {
        test_invariants(
            &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
            |MulPair(m1, m2)| &m1 * &m2,
        );
    }

    #[test]
    fn transpose() {
        test_invariants(&DokMatrix::<i8>::arb_matrix(), DokMatrix::transpose);
    }

    #[test]
    fn set_element() {
        test_invariants(
            &(1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(|(rows, cols)| {
                (
                    DokMatrix::<i8>::arb_fixed_size_matrix(
                        rows.try_into().unwrap(),
                        cols.try_into().unwrap(),
                    ),
                    0..rows,
                    0..cols,
                    any::<i8>(),
                )
            }),
            |(mut m, i, j, t)| {
                m.set_element((i, j), t).unwrap();
                m
            },
        );
    }
}

mod csr {
    #[cfg(feature = "mkl")]
    use {crate::ComplexNewtype, std::convert::TryFrom};

    use std::{convert::TryInto, fmt::Debug, num::Wrapping};

    use cap_rand::{ambient_authority, prelude::CapRng};
    use itertools::iproduct;
    use num::Num;
    use proptest::{
        arbitrary::any, prop_assert, prop_assert_eq, strategy::Strategy, test_runner::TestRunner,
    };

    use crate::{
        csr_matrix::CsrMatrix,
        dok_matrix::DokMatrix,
        proptest::{arb_add_pair, arb_mul_pair},
        AddPair, Matrix, MulPair,
    };

    const MAX_SIZE: usize = 10;

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
        let dok_val = |(mut m, i, j, t): (DokMatrix<_>, _, _, _)| {
            m.set_element((i, j), t).unwrap();
            m
        };
        test_commutes(
            &strategy,
            dok_val,
            |(m, i, j, t)| {
                let ambient_authority = ambient_authority();
                let mut rng = CapRng::default(ambient_authority);
                let mut m1 = CsrMatrix::from_dok(m, &mut rng);
                m1.set_element((i, j), t).unwrap();
                m1
            },
            DokMatrix::from,
        );
        test_commutes(
            &strategy,
            dok_val,
            |(m, i, j, t)| {
                let mut m1 = CsrMatrix::from(m);
                m1.set_element((i, j), t).unwrap();
                m1
            },
            DokMatrix::from,
        );
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

    #[cfg(feature = "mkl")]
    #[test]
    fn mkl_spmm_d() {
        use crate::csr_matrix::mkl::{MklCsrMatrix, RustMklSparseMatrix};

        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair(CsrMatrix::<f64, false>::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let mut m3 = MklCsrMatrix::try_from(m1).unwrap();
                    let m3 = RustMklSparseMatrix::try_from(&mut m3).unwrap();
                    let mut m4 = MklCsrMatrix::try_from(m2).unwrap();
                    let m4 = RustMklSparseMatrix::try_from(&mut m4).unwrap();
                    let m5 = CsrMatrix::try_from((&m3 * &m4).unwrap()).unwrap();
                    prop_assert!(m5.invariants(), "{:?}", m5);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[cfg(feature = "mkl")]
    #[test]
    fn mkl_spmm_z() {
        use crate::csr_matrix::mkl::{MklCsrMatrix, RustMklSparseMatrix};

        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair(CsrMatrix::<ComplexNewtype<f64>, false>::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let mut m3 = MklCsrMatrix::try_from(m1).unwrap();
                    let m3 = RustMklSparseMatrix::try_from(&mut m3).unwrap();
                    let mut m4 = MklCsrMatrix::try_from(m2).unwrap();
                    let m4 = RustMklSparseMatrix::try_from(&mut m4).unwrap();
                    let m5 = CsrMatrix::try_from((&m3 * &m4).unwrap()).unwrap();
                    prop_assert!(m5.invariants(), "{:?}", m5);
                    Ok(())
                },
            )
            .unwrap();
    }
}
