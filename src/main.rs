use quickcheck::{Arbitrary, StdGen};
use rand::thread_rng;
use sparse_matrix::{AddPair, CSRMatrix, MulPair};

fn main() {
    let mut g = StdGen::new(thread_rng(), 1 << 14);
    let MulPair(m1, m2) = MulPair::<i32>::arbitrary(&mut g);
    let (mut m1, m2) = (CSRMatrix::from(m1), CSRMatrix::from(m2));
    m1 *= &m2;
}
