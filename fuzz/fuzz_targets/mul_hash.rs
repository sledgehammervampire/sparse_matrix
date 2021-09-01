#![no_main]
use std::convert::TryInto;

use cap_rand::prelude::CapRng;
use cap_std::ambient_authority;
use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam_csr::CsrMatrix;
use spam_dok::{into_float_matrix_market, DokMatrix};
use spam_matrix::{arbitrary::arb_mul_pair_fixed_size, Matrix, MulPair};

fuzz_target!(|bytes| {
    let ambient_authority = ambient_authority();
    let mut rng = CapRng::default(ambient_authority);
    let mut u = Unstructured::new(bytes);
    if let (Ok(l), Ok(m), Ok(n)) = (
        u.int_in_range(1..=1 << 8),
        u.int_in_range(1..=1 << 8),
        u.int_in_range(1..=(1 << 32 - 1)),
    ) {
        if let Ok(MulPair(m1, m2)) = arb_mul_pair_fixed_size::<f64, DokMatrix<f64>>(
            &mut u,
            l.try_into().unwrap(),
            m.try_into().unwrap(),
            n.try_into().unwrap(),
        ) {
            let m3: CsrMatrix<_, false> = CsrMatrix::from_dok(m1.clone(), &mut rng);
            let m4: CsrMatrix<_, false> = CsrMatrix::from_dok(m2.clone(), &mut rng);
            let m5: CsrMatrix<_, false> = m3.mul_hash(&m4);
            assert!(m5.invariants());
            if m3
                .rows()
                .get()
                .checked_mul(m3.cols().get())
                .and_then(|x| x.checked_mul(m4.cols().get()))
                .map_or(false, |x| x < (1 << 15))
            {
                let m3 = DokMatrix::from(m3);
                let m4 = DokMatrix::from(m4);
                let m5 = DokMatrix::from(m5);
                if let Ok(b) = m5.is_good_approx_of_mul(&m3, &m4) {
                    let mut s3 = String::new();
                    into_float_matrix_market(m3, &mut s3).unwrap();
                    let mut s4 = String::new();
                    into_float_matrix_market(m4, &mut s4).unwrap();
                    assert!(b, "{}\n{}", s3, s4);
                }
            }
        }
    }
});
