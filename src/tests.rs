mod dok {
    use std::{convert::TryInto, fs, num::Wrapping};

    use cap_rand::{ambient_authority, prelude::CapRng};
    use num::Num;
    use proptest::{prop_assert, strategy::Strategy, test_runner::TestRunner};
    use walkdir::WalkDir;

    use crate::{
        csr_matrix::CsrMatrix,
        dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
        proptest::{arb_add_pair, arb_mul_pair},
        AddPair, Matrix, MulPair,
    };

    const MAX_SIZE: usize = 10;

    fn constructor_invariants<S, F, T>(strategy: &S, f: F)
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
        constructor_invariants(
            &(1..MAX_SIZE, 1..MAX_SIZE)
                .prop_map(|(m, n)| (m.try_into().unwrap(), n.try_into().unwrap())),
            DokMatrix::<i8>::new,
        );
    }

    #[test]
    fn identity_invariants() {
        constructor_invariants(
            &(1..MAX_SIZE).prop_map(|n| n.try_into().unwrap()),
            DokMatrix::<i8>::identity,
        );
    }

    #[test]
    fn arb_invariants() {
        constructor_invariants(&DokMatrix::<i8>::arb_matrix(), |m| m);
    }

    #[test]
    fn from_arb_csr_invariants() {
        constructor_invariants(&CsrMatrix::<i8, false>::arb_matrix(), DokMatrix::from);
        constructor_invariants(&CsrMatrix::<i8, true>::arb_matrix(), DokMatrix::from);
    }

    #[ignore = "expensive, parsing code not changed often"]
    #[test]
    fn from_matrix_market() {
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
                            let m1: CsrMatrix<_, false> = CsrMatrix::from_dok(m.clone(), &mut rng);
                            assert!(m1.invariants());
                            let m1: CsrMatrix<_, true> = CsrMatrix::from(m.clone());
                            assert!(m1.invariants());
                        }
                        MatrixType::Real(m) => {
                            let m1: CsrMatrix<_, false> = CsrMatrix::from_dok(m.clone(), &mut rng);
                            assert!(m1.invariants());
                            let m1: CsrMatrix<_, true> = CsrMatrix::from(m.clone());
                            assert!(m1.invariants());
                        }
                        MatrixType::Complex(m) => {
                            let m1: CsrMatrix<_, false> = CsrMatrix::from_dok(m.clone(), &mut rng);
                            assert!(m1.invariants());
                            let m1: CsrMatrix<_, true> = CsrMatrix::from(m.clone());
                            assert!(m1.invariants());
                        }
                    };
                }
            }
        }
    }

    // inductive cases
    #[test]
    fn add() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |AddPair(m1, m2)| {
                    let m = m1.clone() + m2.clone();
                    prop_assert!(m.invariants());
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn mul() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let m = &m1 * &m2;
                    prop_assert!(m.invariants());
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn transpose() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = m.clone().transpose();
                prop_assert!(m1.invariants());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn set_element() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_add_pair(DokMatrix::<i8>::arb_fixed_size_matrix),
                |AddPair(mut m1, m2)| {
                    for (pos, t) in m2.iter() {
                        m1.set_element(pos, t.clone()).unwrap();
                    }
                    prop_assert!(m1.invariants());
                    Ok(())
                },
            )
            .unwrap();
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

    fn test_constructor<S, F, G, H, T>(strategy: &S, f: F, g: G, h: H)
    where
        S: Strategy,
        S::Value: Clone,
        T: Num + Debug,
        F: Fn(S::Value) -> DokMatrix<T>,
        G: Fn(S::Value) -> CsrMatrix<T, false>,
        H: Fn(S::Value) -> CsrMatrix<T, true>,
    {
        let mut runner = TestRunner::default();
        runner
            .run(strategy, |v| {
                let m = f(v.clone());
                let m1 = g(v.clone());
                prop_assert!(m1.invariants());
                prop_assert_eq!(&DokMatrix::from(m1), &m);
                let m1 = h(v);
                prop_assert!(m1.invariants());
                prop_assert_eq!(DokMatrix::from(m1), m);
                Ok(())
            })
            .unwrap();
    }

    // base cases
    #[test]
    fn new() {
        test_constructor(
            &(1..MAX_SIZE, 1..MAX_SIZE)
                .prop_map(|(m, n)| (m.try_into().unwrap(), n.try_into().unwrap())),
            DokMatrix::<i8>::new,
            CsrMatrix::new,
            CsrMatrix::new,
        );
    }

    #[test]
    fn identity_invariants() {
        test_constructor(
            &(1..MAX_SIZE).prop_map(|m| m.try_into().unwrap()),
            DokMatrix::<i8>::identity,
            CsrMatrix::identity,
            CsrMatrix::identity,
        );
    }

    #[test]
    fn arb_invariants() {
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
    fn from_arb_dok_invariants() {
        test_constructor(
            &DokMatrix::<i8>::arb_matrix(),
            |m| m,
            |m| {
                let ambient_authority = ambient_authority();
                let mut rng = CapRng::default(ambient_authority);
                CsrMatrix::from_dok(m, &mut rng)
            },
            CsrMatrix::from,
        );
    }

    // other
    #[test]
    fn rows() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let ambient_authority = ambient_authority();
                let mut rng = CapRng::default(ambient_authority);

                let m1 = CsrMatrix::from_dok(m.clone(), &mut rng);
                prop_assert!(m.rows() == m1.rows());
                let m1 = CsrMatrix::from(m.clone());
                prop_assert!(m.rows() == m1.rows());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn cols() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let ambient_authority = ambient_authority();
                let mut rng = CapRng::default(ambient_authority);

                let m1 = CsrMatrix::from_dok(m.clone(), &mut rng);
                prop_assert!(m.cols() == m1.cols());
                let m1 = CsrMatrix::from(m.clone());
                prop_assert!(m.cols() == m1.cols());
                Ok(())
            })
            .unwrap();
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
        let mut runner = TestRunner::default();
        runner
            .run(
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
                    let ambient_authority = ambient_authority();
                    let mut rng = CapRng::default(ambient_authority);

                    let mut m1 = CsrMatrix::from_dok(m.clone(), &mut rng);
                    m.set_element((i, j), t.clone()).unwrap();
                    m1.set_element((i, j), t.clone()).unwrap();
                    prop_assert!(m1.invariants(), "{:?}", m1);
                    prop_assert_eq!(&DokMatrix::from(m1), &m);
                    let mut m1 = CsrMatrix::from(m.clone());
                    m.set_element((i, j), t.clone()).unwrap();
                    m1.set_element((i, j), t.clone()).unwrap();
                    prop_assert!(m1.invariants(), "{:?}", m1);
                    prop_assert_eq!(DokMatrix::from(m1), m);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn transpose() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let ambient_authority = ambient_authority();
                let mut rng = CapRng::default(ambient_authority);

                let m1 = CsrMatrix::from_dok(m.clone(), &mut rng).transpose();
                prop_assert!(m1.invariants(), "{:?}", m1);
                prop_assert_eq!(DokMatrix::from(m1), m.clone().transpose());
                let m1 = CsrMatrix::from(m.clone()).transpose();
                prop_assert!(m1.invariants(), "{:?}", m1);
                prop_assert_eq!(DokMatrix::from(m1), m.transpose());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn add() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |AddPair(m1, m2)| {
                    let ambient_authority = ambient_authority();
                    let mut rng = CapRng::default(ambient_authority);

                    let m = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        + CsrMatrix::from_dok(m2.clone(), &mut rng);
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), m1.clone() + m2.clone());
                    let m = CsrMatrix::from(m1.clone()) + CsrMatrix::from(m2.clone());
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), m1 + m2);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn mul() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let ambient_authority = ambient_authority();
                    let mut rng = CapRng::default(ambient_authority);

                    let m = &CsrMatrix::from_dok(m1.clone(), &mut rng)
                        * &CsrMatrix::from_dok(m2.clone(), &mut rng);
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m = &CsrMatrix::from(m1.clone()) * &CsrMatrix::from(m2.clone());
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn mul_hash() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let ambient_authority = ambient_authority();
                    let mut rng = CapRng::default(ambient_authority);

                    let m: CsrMatrix<_, false> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, false> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, false> = CsrMatrix::from(m1.clone())
                        .mul_hash(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, false> =
                        CsrMatrix::from(m1.clone()).mul_hash(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> = CsrMatrix::from(m1.clone())
                        .mul_hash(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> =
                        CsrMatrix::from(m1.clone()).mul_hash(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn mul_btree() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let ambient_authority = ambient_authority();
                    let mut rng = CapRng::default(ambient_authority);

                    let m = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_btree(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_btree(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m = CsrMatrix::from(m1.clone())
                        .mul_btree(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m = CsrMatrix::from(m1.clone()).mul_btree(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn mul_heap() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let m = CsrMatrix::from(m1.clone()).mul_heap(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    Ok(())
                },
            )
            .unwrap();
    }

    // #[test]
    // fn mul_esc() {
    //     let mut runner = TestRunner::default();
    //     runner
    //         .run(
    //             &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
    //             |MulPair(m1, m2)| {
    //                 let m = <CsrMatrix<_, false> as From<_>>::from(m1.clone())
    //                     .mul_esc::<false>(&CsrMatrix::from(m2.clone()));
    //                 prop_assert!(m.invariants(), "{:?}", m);
    //                 prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
    //                 let m = <CsrMatrix<_, false> as From<_>>::from(m1.clone())
    //                     .mul_esc::<true>(&CsrMatrix::from(m2.clone()));
    //                 prop_assert!(m.invariants(), "{:?}", m);
    //                 prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
    //                 let m = <CsrMatrix<_, true> as From<_>>::from(m1.clone())
    //                     .mul_esc::<false>(&CsrMatrix::from(m2.clone()));
    //                 prop_assert!(m.invariants(), "{:?}", m);
    //                 prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
    //                 let m = <CsrMatrix<_, true> as From<_>>::from(m1.clone())
    //                     .mul_esc::<true>(&CsrMatrix::from(m2.clone()));
    //                 prop_assert!(m.invariants(), "{:?}", m);
    //                 prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
    //                 Ok(())
    //             },
    //         )
    //         .unwrap();
    // }

    #[test]
    fn mul_hash2() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let ambient_authority = ambient_authority();
                    let mut rng = CapRng::default(ambient_authority);

                    let m: CsrMatrix<_, false> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash2(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, false> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash2(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, false> = CsrMatrix::from(m1.clone())
                        .mul_hash2(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, false> =
                        CsrMatrix::from(m1.clone()).mul_hash2(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash2(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> = CsrMatrix::from_dok(m1.clone(), &mut rng)
                        .mul_hash2(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> = CsrMatrix::from(m1.clone())
                        .mul_hash2(&CsrMatrix::from_dok(m2.clone(), &mut rng));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    let m: CsrMatrix<_, true> =
                        CsrMatrix::from(m1.clone()).mul_hash2(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(DokMatrix::from(m), &m1 * &m2);
                    Ok(())
                },
            )
            .unwrap();
    }

    #[test]
    fn dok_roundtrip() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let ambient_authority = ambient_authority();
                let mut rng = CapRng::default(ambient_authority);

                prop_assert_eq!(
                    &m,
                    &DokMatrix::from(CsrMatrix::from_dok(m.clone(), &mut rng))
                );
                prop_assert_eq!(&m, &DokMatrix::from(CsrMatrix::from(m.clone())));
                Ok(())
            })
            .unwrap();
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
                    let mut m3 = MklCsrMatrix::try_from(m1.clone()).unwrap();
                    let m3 = RustMklSparseMatrix::try_from(&mut m3).unwrap();
                    let mut m4 = MklCsrMatrix::try_from(m2.clone()).unwrap();
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
                    let mut m3 = MklCsrMatrix::try_from(m1.clone()).unwrap();
                    let m3 = RustMklSparseMatrix::try_from(&mut m3).unwrap();
                    let mut m4 = MklCsrMatrix::try_from(m2.clone()).unwrap();
                    let m4 = RustMklSparseMatrix::try_from(&mut m4).unwrap();
                    let m5 = CsrMatrix::try_from((&m3 * &m4).unwrap()).unwrap();
                    prop_assert!(m5.invariants(), "{:?}", m5);
                    Ok(())
                },
            )
            .unwrap();
    }
}
