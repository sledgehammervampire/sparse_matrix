#[cfg(feature = "debug")]
use std::collections::BTreeMap;
use std::{
    alloc::{Allocator, Global},
    hash::{BuildHasher, BuildHasherDefault},
    mem,
};

use crate::{MulHasher, MIN_TABLE_SIZE};

pub struct HashSet<S = BuildHasherDefault<MulHasher>, A = Global>
where
    A: Allocator,
{
    hash_builder: S,
    slots: Vec<u32, A>,
    upper_bound: usize,
    items: usize,
    #[cfg(feature = "debug")]
    pub probe_lengths: BTreeMap<usize, usize>,
}

impl HashSet {
    pub fn new() -> Self {
        HashSet::with_capacity(MIN_TABLE_SIZE / 4)
    }
    pub fn with_capacity(capacity: usize) -> Self {
        HashSet::with_capacity_and_hasher_in(capacity, Global, BuildHasherDefault::default())
    }
}
impl<S> HashSet<S> {
    pub fn with_hasher(hash_builder: S) -> HashSet<S, Global> {
        HashSet::with_capacity_and_hasher_in(MIN_TABLE_SIZE / 4, Global, hash_builder)
    }
}
impl<S, A: Allocator> HashSet<S, A> {
    pub fn with_capacity_and_hasher_in(capacity: usize, alloc: A, hash_builder: S) -> Self {
        let upper_bound = capacity
            .checked_next_power_of_two()
            .expect("next power of 2 doesn't fit a usize")
            .checked_mul(2)
            .expect("multiplication by 2 overflows a usize")
            .max(MIN_TABLE_SIZE);
        let mut slots = Vec::with_capacity_in(upper_bound, alloc);
        slots.resize(upper_bound, u32::MAX);
        Self {
            hash_builder,
            slots,
            upper_bound,
            items: 0,
            #[cfg(feature = "debug")]
            probe_lengths: BTreeMap::new(),
        }
    }
    pub fn shrink_to(&mut self, capacity: usize) {
        debug_assert!(self.is_empty());
        let upper_bound = capacity
            .checked_next_power_of_two()
            .expect("next power of 2 doesn't fit in a usize")
            .checked_mul(2)
            .expect("multiplication by 2 overflows a usize")
            .max(MIN_TABLE_SIZE);
        self.upper_bound = upper_bound.min(self.upper_bound);
    }
    pub fn is_empty(&self) -> bool {
        self.items == 0
    }
    pub fn len(&self) -> usize {
        self.items
    }
    pub fn clear(&mut self) {
        self.slots[..self.upper_bound].fill(u32::MAX);
        self.items = 0;
    }
}
impl<A: Allocator + Clone, S: BuildHasher> HashSet<S, A> {
    fn grow(&mut self) {
        if self.upper_bound == self.slots.len() {
            self.slots.resize(
                self.slots
                    .len()
                    .checked_mul(2)
                    .expect("multiplication by 2 overflows a usize"),
                u32::MAX,
            );
        }
        let mut keys = Vec::with_capacity_in(self.items, self.slots.allocator().clone());
        keys.extend(
            self.slots[..self.upper_bound]
                .iter_mut()
                .filter_map(|key| (*key != u32::MAX).then(|| mem::replace(key, u32::MAX))),
        );
        self.upper_bound = self
            .upper_bound
            .checked_mul(2)
            .expect("multiplication by 2 overflows a usize");
        for key in keys {
            let hash = self.hash_builder.hash_one(key);
            insert_raw(
                &mut self.slots[..self.upper_bound],
                key,
                hash,
                #[cfg(feature = "debug")]
                &mut self.probe_lengths,
            );
        }
    }
    #[inline]
    pub fn insert(&mut self, key: u32) {
        debug_assert!(key != u32::MAX);
        let hash = self.hash_builder.hash_one(key);
        if insert_raw(
            &mut self.slots[..self.upper_bound],
            key,
            hash,
            #[cfg(feature = "debug")]
            &mut self.probe_lengths,
        ) {
            self.items += 1;
        }
        if self.items > self.upper_bound / 2 {
            self.grow();
        }
    }
}

#[inline]
fn insert_raw(
    slots: &mut [u32],
    key: u32,
    hash: u64,
    #[cfg(feature = "debug")] probe_lengths: &mut BTreeMap<usize, usize>,
) -> bool {
    let mut index = hash as usize & (slots.len() - 1);
    #[cfg(feature = "debug")]
    let mut probes = 0;
    loop {
        let curr = &mut slots[index];
        if *curr == key {
            #[cfg(feature = "debug")]
            {
                *probe_lengths.entry(probes).or_insert(0) += 1;
            }
            break false;
        } else if *curr == u32::MAX {
            #[cfg(feature = "debug")]
            {
                *probe_lengths.entry(probes).or_insert(0) += 1;
            }
            *curr = key;
            break true;
        } else {
            #[cfg(feature = "debug")]
            {
                probes += 1;
            }
            index = (index + 1) & (slots.len() - 1);
        }
    }
}

impl Default for HashSet {
    fn default() -> Self {
        HashSet::new()
    }
}
