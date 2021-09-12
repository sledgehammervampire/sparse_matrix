#![feature(allocator_api)]
#![deny(clippy::disallowed_method)]

#[cfg(feature = "debug")]
use std::collections::BTreeMap;
use std::{
    alloc::{Allocator, Global},
    mem,
    num::NonZeroU8,
};

const HASH_SCAL: usize = 107;
const MIN_TABLE_SIZE: usize = 16;

pub struct HashSet<A = Global>
where
    A: Allocator,
{
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
        HashSet::with_capacity_in(capacity, Global)
    }
}
impl<A: Allocator + Clone> HashSet<A> {
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let upper_bound = capacity
            .checked_next_power_of_two()
            .expect("next power of 2 doesn't fit a usize")
            .checked_mul(2)
            .expect("multiplication by 2 overflows a usize")
            .max(MIN_TABLE_SIZE);
        let mut slots = Vec::with_capacity_in(upper_bound, alloc);
        slots.resize(upper_bound, u32::MAX);
        Self {
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
            insert_raw(
                &mut self.slots[..self.upper_bound],
                key,
                #[cfg(feature = "debug")]
                &mut self.probe_lengths,
            );
        }
    }
    #[inline]
    pub fn insert(&mut self, key: u32) {
        debug_assert!(key != u32::MAX);
        if insert_raw(
            &mut self.slots[..self.upper_bound],
            key,
            #[cfg(feature = "debug")]
            &mut self.probe_lengths,
        ) {
            self.items += 1;
        }
        if self.items > self.upper_bound / 2 {
            self.grow();
        }
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

#[inline]
fn insert_raw(
    slots: &mut [u32],
    key: u32,
    #[cfg(feature = "debug")] probe_lengths: &mut BTreeMap<usize, usize>,
) -> bool {
    let mut index = usize::try_from(key).unwrap().wrapping_mul(HASH_SCAL) & (slots.len() - 1);
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

pub struct HashMap<K, V, A = Global>
where
    A: Allocator,
{
    // size_of::<Option<(NonZeroU8,u32,f64)>> == 16 while size_of::<Option<(u32,f64)>> == 24
    slots: Vec<Option<(NonZeroU8, K, V)>, A>,
    capacity: usize,
    #[cfg(feature = "debug")]
    pub probe_lengths: BTreeMap<usize, usize>,
}

impl<V: Copy> HashMap<u32, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        HashMap::with_capacity_in(capacity, Global)
    }
}

impl<V: Copy, A: Allocator> HashMap<u32, V, A> {
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        let capacity = capacity
            .checked_next_power_of_two()
            .expect("next power of 2 doesn't fit a usize")
            .checked_mul(2)
            .expect("multiplication by 2 overflows a usize")
            .max(MIN_TABLE_SIZE);
        let mut slots = Vec::with_capacity_in(capacity, alloc);
        slots.resize(capacity, None);
        Self {
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
    pub fn entry(&mut self, key: u32) -> Entry<'_, u32, V> {
        debug_assert!(key != u32::MAX);
        let mut hash = usize::try_from(key).unwrap().wrapping_mul(HASH_SCAL) & (self.capacity - 1);
        #[cfg(feature = "debug")]
        let mut probes = 0;
        loop {
            match self.slots[hash] {
                Some((_, k, _)) if k == key => {
                    #[cfg(feature = "debug")]
                    {
                        *self.probe_lengths.entry(probes).or_insert(0) += 1;
                    }
                    break Entry::Occupied(&mut self.slots[hash].as_mut().unwrap().2);
                }
                None => {
                    #[cfg(feature = "debug")]
                    {
                        *self.probe_lengths.entry(probes).or_insert(0) += 1;
                    }
                    break Entry::Vacant(key, &mut self.slots[hash]);
                }
                Some(_) => {
                    #[cfg(feature = "debug")]
                    {
                        probes += 1;
                    }
                    hash = (hash + 1) & (self.capacity - 1)
                }
            }
        }
    }
    pub fn drain(&mut self) -> impl Iterator<Item = (u32, V)> + '_ {
        self.slots[..self.capacity]
            .iter_mut()
            .filter_map(|e| e.take().map(|(_, k, v)| (k, v)))
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
