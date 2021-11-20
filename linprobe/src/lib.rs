#![feature(allocator_api)]
#![feature(build_hasher_simple_hash_one)]
#![deny(clippy::disallowed_method)]

use std::hash::Hasher;

mod map;
mod set;

pub use map::HashMap;
pub use set::HashSet;

const HASH_SCAL: u32 = 107;
const MIN_TABLE_SIZE: usize = 16;

#[derive(Default)]
pub struct MulHasher(u32);
impl Hasher for MulHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0 as u64
    }

    fn write(&mut self, _: &[u8]) {
        todo!()
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = i.wrapping_mul(HASH_SCAL);
    }
}
