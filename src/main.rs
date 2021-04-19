use spam::{csr_matrix::CsrMatrix, dok_matrix::parse_matrix_market};
use std::fs::read_to_string;

fn main() {
    let f = "engine";
    let m1 = CsrMatrix::from(
        parse_matrix_market::<i32>(&read_to_string(format!("matrices/{}.mtx", f)).unwrap())
            .unwrap()
            .1,
    );
    let m2 = m1.clone();
    let _ = &m1 * &m2;
}
