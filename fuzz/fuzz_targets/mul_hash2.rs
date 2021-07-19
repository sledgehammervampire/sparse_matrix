#![no_main]
use cap_rand::prelude::CapRng;
use cap_std::ambient_authority;
use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam::{
    arbitrary::arb_mul_pair_fixed_size, csr_matrix::CsrMatrix, dok_matrix::DokMatrix, MulPair,
};

fuzz_target!(|bytes| {
    const MAX_SIZE: usize = 100;

    let ambient_authority = ambient_authority();
    let mut rng = CapRng::default(ambient_authority);
    let mut u = Unstructured::new(bytes);
    if let (Ok(l), Ok(m), Ok(n)) = (
        u.int_in_range(0..=MAX_SIZE),
        u.int_in_range(0..=MAX_SIZE),
        u.int_in_range(0..=MAX_SIZE),
    ) {
        if let Ok(Ok(MulPair(m1, m2))) =
            arb_mul_pair_fixed_size::<f64, DokMatrix<f64>>(&mut u, l, m, n)
        {
            let m3: CsrMatrix<_, false> = CsrMatrix::from_dok(m1.clone(), &mut rng);
            let m4: CsrMatrix<_, false> = CsrMatrix::from_dok(m2.clone(), &mut rng);
            let m5: CsrMatrix<_, false> = m3.mul_hash2(&m4);
            assert!(m5.invariants());
            let m6 = &m1 * &m2;
            if m5.iter().all(|(_, t)| !t.is_nan()) && m6.iter().all(|(_, t)| !t.is_nan()) {
                assert_eq!(DokMatrix::from(m5), m6);
            }
        }
    }
});