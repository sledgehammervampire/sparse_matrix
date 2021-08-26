#![no_main]
use std::convert::TryInto;

use cap_rand::prelude::CapRng;
use cap_std::ambient_authority;
use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use num::Float;
use sci::Sci;
use spam_csr::CsrMatrix;
use spam_dok::DokMatrix;
use spam_matrix::{arbitrary::arb_mul_pair_fixed_size, Matrix, MulPair};

fuzz_target!(|bytes| {
    const MAX_SIZE: usize = 10;

    let ambient_authority = ambient_authority();
    let mut rng = CapRng::default(ambient_authority);
    let mut u = Unstructured::new(bytes);
    if let (Ok(l), Ok(m), Ok(n)) = (
        u.int_in_range(1..=MAX_SIZE),
        u.int_in_range(1..=MAX_SIZE),
        u.int_in_range(1..=MAX_SIZE),
    ) {
        if let Ok(MulPair(m1, m2)) = arb_mul_pair_fixed_size::<Sci<f64>, DokMatrix<Sci<f64>>>(
            &mut u,
            l.try_into().unwrap(),
            m.try_into().unwrap(),
            n.try_into().unwrap(),
        ) {
            let m3: CsrMatrix<_, false> = CsrMatrix::from_dok(m1.clone(), &mut rng);
            let m4: CsrMatrix<_, false> = CsrMatrix::from_dok(m2.clone(), &mut rng);
            let m5: CsrMatrix<_, false> = m3.mul_hash(&m4);
            assert!(m5.invariants());
            let m6 = &m1 * &m2;
            let m5 = DokMatrix::from(m5);
            let all_not_nan = |m: &DokMatrix<Sci<f64>>| m.iter().all(|(_, t)| !t.is_nan());
            if all_not_nan(&m5) && all_not_nan(&m6) {
                assert!(m5.approx_eq(&m6), "{:?}", m5 - m6);
            }
        }
    }
});
