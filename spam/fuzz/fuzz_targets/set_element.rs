#![no_main]

use libfuzzer_sys::{arbitrary::Unstructured, fuzz_target};
use spam::{csr_matrix::CsrMatrix, dok_matrix::DokMatrix, Matrix};

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    match u.arbitrary::<DokMatrix<i8>>() {
        Ok(mut m) if m.rows() > 0 && m.cols() > 0 => {
            if let (Ok(i), Ok(j), Ok(t)) = (
                u.int_in_range(0..=m.rows() - 1),
                u.int_in_range(0..=m.cols() - 1),
                u.arbitrary::<i8>(),
            ) {
                let mut m1 = CsrMatrix::from(m.clone());
                m1.set_element((i, j), t.clone());
                m.set_element((i, j), t.clone());
                assert!(m1.invariants());
                assert_eq!(m1, CsrMatrix::from(m));
            }
        }
        _ => {}
    }
});
