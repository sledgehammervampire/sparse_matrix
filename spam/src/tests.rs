mod dok {
    use std::{fs, num::Wrapping};

    use proptest::{prop_assert, prop_assert_eq, test_runner::TestRunner};
    use walkdir::WalkDir;

    use crate::{
        csr_matrix::CsrMatrix,
        dok_matrix::{parse_matrix_market, DokMatrix, MatrixType},
        proptest::{arb_add_pair, arb_mul_pair},
        AddPair, Matrix, MulPair,
    };

    const MAX_SIZE: usize = 100;

    // base cases
    #[test]
    fn identity_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&(0..MAX_SIZE), |n| {
                if let Ok(m) = DokMatrix::<i8>::identity(n) {
                    prop_assert!(m.invariants());
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn arb_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                prop_assert!(m.invariants());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn from_arb_csr_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&CsrMatrix::<i8>::arb_matrix(), |m| {
                let m = DokMatrix::from(m);
                prop_assert!(m.invariants());
                Ok(())
            })
            .unwrap();
    }

    #[ignore = "expensive, parsing code not changed often"]
    #[test]
    fn from_matrix_market() {
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
                            let m = CsrMatrix::from(m);
                            assert!(m.invariants());
                        }
                        MatrixType::Real(m) => {
                            let m = CsrMatrix::from(m);
                            assert!(m.invariants());
                        }
                        MatrixType::Complex(m) => {
                            let m = CsrMatrix::from(m);
                            assert!(m.invariants());
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
                    for (pos, t) in m2.entries() {
                        m1.set_element(pos, t.clone());
                    }
                    prop_assert!(m1.invariants());
                    Ok(())
                },
            )
            .unwrap();
    }

    // other
    #[test]
    fn convert() {
        let mut runner = TestRunner::default();
        runner
            .run(&CsrMatrix::<i8>::arb_matrix(), |m| {
                prop_assert_eq!(&m, &CsrMatrix::from(DokMatrix::from(m.clone())));
                Ok(())
            })
            .unwrap();
    }
}

mod csr {
    use std::{convert::TryFrom, num::Wrapping};

    use itertools::iproduct;
    use proptest::{
        arbitrary::any, prop_assert, prop_assert_eq, strategy::Strategy, test_runner::TestRunner,
    };

    use crate::{
        csr_matrix::{
            ffi::{CMklSparseMatrix, MklCsrMatrix, RustMklSparseMatrix},
            CsrMatrix,
        },
        dok_matrix::DokMatrix,
        proptest::{arb_add_pair, arb_mul_pair},
        AddPair, Matrix, MulPair,
    };

    const MAX_SIZE: usize = 100;

    // base cases
    #[test]
    fn new_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&(0..MAX_SIZE, 0..MAX_SIZE), |(m, n)| {
                if let Ok(m1) = CsrMatrix::<i8>::new(m, n) {
                    prop_assert!(m1.invariants());
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn identity_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&(0..MAX_SIZE), |n| {
                if let Ok(m) = CsrMatrix::<i8>::identity(n) {
                    prop_assert!(m.invariants());
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn arb_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&CsrMatrix::<i8>::arb_matrix(), |m| {
                prop_assert!(m.invariants(), "{:?}", m);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn from_arb_dok_invariants() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m = CsrMatrix::from(m);
                prop_assert!(m.invariants(), "{:?}", m);
                Ok(())
            })
            .unwrap();
    }

    // inductive cases
    #[test]
    fn add() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_add_pair::<Wrapping<i8>, _, _>(DokMatrix::arb_fixed_size_matrix),
                |AddPair(m1, m2)| {
                    let m = CsrMatrix::from(m1.clone()) + CsrMatrix::from(m2.clone());
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(m, CsrMatrix::from(m1 + m2));
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
                    let m = &CsrMatrix::from(m1.clone()) * &CsrMatrix::from(m2.clone());
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
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
                    let m = CsrMatrix::from(m1.clone()).mul_hash(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
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
                    let m = CsrMatrix::from(m1.clone()).mul_btree(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
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
                    let m = CsrMatrix::from(m1.clone()).mul_hash(&CsrMatrix::from(m2.clone()));
                    prop_assert!(m.invariants(), "{:?}", m);
                    prop_assert_eq!(m, CsrMatrix::from(&m1 * &m2));
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
                let m1 = CsrMatrix::from(m.clone()).transpose();
                prop_assert!(m1.invariants(), "{:?}", m1);
                prop_assert_eq!(m1, CsrMatrix::from(m.transpose()));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn set_element() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &(1..MAX_SIZE, 1..MAX_SIZE).prop_flat_map(|(rows, cols)| {
                    (
                        DokMatrix::<i8>::arb_fixed_size_matrix(rows, cols),
                        0..rows,
                        0..cols,
                        any::<i8>(),
                    )
                }),
                |(mut m, i, j, t)| {
                    let mut m1 = CsrMatrix::from(m.clone());
                    m.set_element((i, j), t.clone());
                    m1.set_element((i, j), t.clone());
                    assert!(m1.invariants(), "{:?}", m1);
                    assert_eq!(m1, CsrMatrix::from(m));
                    Ok(())
                },
            )
            .unwrap();
    }

    // other
    #[test]
    fn iter() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = CsrMatrix::from(m.clone());
                prop_assert!(m1.iter().eq(m.entries()));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn rows_and_cols() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = CsrMatrix::from(m.clone());
                prop_assert_eq!((m.rows(), m.cols()), (m1.rows(), m1.cols()));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn get_element() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                let m1 = CsrMatrix::from(m.clone());
                prop_assert!(iproduct!(0..m.rows(), 0..m.cols())
                    .all(|pos| m.get_element(pos) == m1.get_element(pos)));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn dok_roundtrip() {
        let mut runner = TestRunner::default();
        runner
            .run(&DokMatrix::<i8>::arb_matrix(), |m| {
                prop_assert_eq!(&m, &DokMatrix::from(CsrMatrix::from(m.clone())));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn mkl_spmm() {
        let mut runner = TestRunner::default();
        runner
            .run(
                &arb_mul_pair(CsrMatrix::<f64>::arb_fixed_size_matrix),
                |MulPair(m1, m2)| {
                    let mut m3 = MklCsrMatrix::try_from(m1.clone()).unwrap();
                    let m3 =
                        CMklSparseMatrix::from(RustMklSparseMatrix::try_from(&mut m3).unwrap());
                    let mut m4 = MklCsrMatrix::try_from(m2.clone()).unwrap();
                    let m4 =
                        CMklSparseMatrix::from(RustMklSparseMatrix::try_from(&mut m4).unwrap());
                    CsrMatrix::try_from((&m3 * &m4).unwrap()).unwrap();
                    Ok(())
                },
            )
            .unwrap();
    }
}
