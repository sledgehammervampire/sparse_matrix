use spam::{csr_matrix::CsrMatrix, dok_matrix::parse_matrix_market};
use std::{env::args, fs::read_to_string};

fn main() {
    let path = args().nth(1).unwrap();
    let m1 = CsrMatrix::from(
        parse_matrix_market::<i32>(&read_to_string(path).unwrap())
            .unwrap()
            .1,
    );
    let m2 = m1.clone();
    let _ = &m1 * &m2;
}
