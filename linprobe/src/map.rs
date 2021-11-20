use std::{
    alloc::{Allocator, Global},
    hash::{BuildHasher, BuildHasherDefault, Hash},
    num::NonZeroU8,
};

use crate::{MulHasher, MIN_TABLE_SIZE};

pub struct HashMap<K, V, S = BuildHasherDefault<MulHasher>, A = Global>
where
    A: Allocator,
{
    hash_builder: S,
    // size_of::<Option<(NonZeroU8,u32,f64)>> == 16 while size_of::<Option<(u32,f64)>> == 24
    slots: Vec<Option<(NonZeroU8, K, V)>, A>,
    capacity: usize,
    #[cfg(feature = "debug")]
    pub probe_lengths: BTreeMap<usize, usize>,
}

impl<K, V> HashMap<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        HashMap::with_capacity_and_hasher_in(capacity, Global, BuildHasherDefault::default())
    }
}
impl<K, V, S> HashMap<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        HashMap::with_capacity_and_hasher_in(capacity, Global, hash_builder)
    }
}
impl<K, V, S, A: Allocator> HashMap<K, V, S, A> {
    pub fn with_capacity_and_hasher_in(capacity: usize, alloc: A, hash_builder: S) -> Self {
        let capacity = capacity
            .checked_next_power_of_two()
            .expect("next power of 2 doesn't fit a usize")
            .checked_mul(2)
            .expect("multiplication by 2 overflows a usize")
            .max(MIN_TABLE_SIZE);
        let mut slots = Vec::with_capacity_in(capacity, alloc);
        slots.resize_with(capacity, || None);
        Self {
            hash_builder,
            slots,
            capacity,
            #[cfg(feature = "debug")]
            probe_lengths: BTreeMap::new(),
        }
    }
    pub fn shrink_to(&mut self, capacity: usize) {
        let capacity = capacity
            .checked_next_power_of_two()
            .expect("next power of 2 doesn't fit in a usize")
            .checked_mul(2)
            .expect("multiplication by 2 overflows a usize")
            .max(MIN_TABLE_SIZE);
        debug_assert!(capacity <= self.slots.len());
        self.capacity = capacity;
    }
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        self.slots[..self.capacity]
            .iter_mut()
            .filter_map(|e| e.take().map(|(_, k, v)| (k, v)))
    }
}

impl<K: Eq + Hash, V, S: BuildHasher, A: Allocator> HashMap<K, V, S, A> {
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        let hash = self.hash_builder.hash_one(&key);
        let mut index = hash as usize & (self.capacity - 1);
        #[cfg(feature = "debug")]
        let mut probes = 0;
        loop {
            match &self.slots[index] {
                Some((_, k, _)) if k == &key => {
                    #[cfg(feature = "debug")]
                    {
                        *self.probe_lengths.entry(probes).or_insert(0) += 1;
                    }
                    break Entry::Occupied(&mut self.slots[index].as_mut().unwrap().2);
                }
                None => {
                    #[cfg(feature = "debug")]
                    {
                        *self.probe_lengths.entry(probes).or_insert(0) += 1;
                    }
                    break Entry::Vacant(key, &mut self.slots[index]);
                }
                Some(_) => {
                    #[cfg(feature = "debug")]
                    {
                        probes += 1;
                    }
                    index = (index + 1) & (self.capacity - 1)
                }
            }
        }
    }
}

pub enum Entry<'a, K, V> {
    Occupied(&'a mut V),
    Vacant(K, &'a mut Option<(NonZeroU8, K, V)>),
}

impl<'a, K, V> Entry<'a, K, V> {
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(ref mut v) = self {
            f(*v);
        }
        self
    }
    pub fn or_insert(self, v: V) -> &'a mut V {
        match self {
            Entry::Occupied(v) => v,
            Entry::Vacant(k, slot) => {
                *slot = Some((NonZeroU8::new(1).unwrap(), k, v));
                slot.as_mut().map(|e| &mut e.2).unwrap()
            }
        }
    }
}
