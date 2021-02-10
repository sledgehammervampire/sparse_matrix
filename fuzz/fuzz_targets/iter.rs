#![no_main]
use libfuzzer_sys::fuzz_target;

use std::collections::HashMap;

use sparse_matrix::{dok_matrix::DokMatrix, CsrMatrix};

fuzz_target!(|m: DokMatrix<i32>| {
    assert_eq!(
        &CsrMatrix::from(m.clone())
            .iter()
            .map(|(p, &t)| (p, t))
            .collect::<HashMap<_, _>>(),
        m.entries()
    );
});
