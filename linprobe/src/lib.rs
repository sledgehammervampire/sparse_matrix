#![feature(allocator_api)]
#![deny(clippy::disallowed_method)]

use std::{
    alloc::{Allocator, Global},
    convert::TryFrom,
    num::NonZeroU8,
};

pub struct HashSet<A = Global>
where
    A: Allocator,
{
    slots: Vec<u32, A>,
    capacity: usize,
    items: usize,
}

impl HashSet {
    pub fn with_capacity(capacity: usize) -> Self {
        HashSet::with_capacity_in(capacity, Global)
    }
}
impl<A: Allocator> HashSet<A> {
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let mut slots = Vec::with_capacity_in(capacity, alloc);
        slots.resize(capacity, u32::MAX);
        Self {
            slots,
            capacity,
            items: 0,
        }
    }
    pub fn shrink_to(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        debug_assert!(capacity <= self.slots.len());
        self.capacity = capacity;
    }
    pub fn insert(&mut self, key: u32) {
        debug_assert!(key != u32::MAX);
        const HASH_SCAL: usize = 107;
        let mut hash = (usize::try_from(key).unwrap() * HASH_SCAL) & (self.capacity - 1);
        loop {
            let curr = &mut self.slots[hash];
            if *curr == key {
                break;
            } else if *curr == u32::MAX {
                *curr = key;
                self.items += 1;
                break;
            } else {
                hash = (hash + 1) & (self.capacity - 1);
            }
        }
    }
    pub fn is_empty(&self) -> bool {
        self.items == 0
    }
    pub fn len(&self) -> usize {
        self.items
    }
    pub fn clear(&mut self) {
        self.slots[..self.capacity].fill(u32::MAX);
        self.items = 0;
    }
}

pub struct HashMap<K, V, A = Global>
where
    A: Allocator,
{
    // size_of::<Option<(NonZeroU8,u32,f64)>> == 16 while size_of::<Option<(u32,f64)>> == 24
    slots: Vec<Option<(NonZeroU8, K, V)>, A>,
    capacity: usize,
    items: usize,
}

impl<V: Copy> HashMap<u32, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        HashMap::with_capacity_in(capacity, Global)
    }
}

impl<V: Copy, A: Allocator> HashMap<u32, V, A> {
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let mut slots = Vec::with_capacity_in(capacity, alloc);
        slots.resize(capacity, None);
        Self {
            slots,
            capacity,
            items: 0,
        }
    }
    // required: self.len() == 0
    pub fn shrink_to(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        debug_assert!(capacity <= self.slots.len());
        self.capacity = capacity;
    }
    pub fn entry(&mut self, key: u32) -> Entry<'_, u32, V> {
        debug_assert!(key != u32::MAX);
        const HASH_SCAL: usize = 107;
        let mut hash = (usize::try_from(key).unwrap() * HASH_SCAL) & (self.capacity - 1);
        loop {
            match self.slots[hash] {
                Some((_, k, _)) if k == key => {
                    break Entry::Occupied(&mut self.slots[hash].as_mut().unwrap().2)
                }
                None => break Entry::Vacant(key, &mut self.slots[hash], &mut self.items),
                Some(_) => hash = (hash + 1) & (self.capacity - 1),
            }
        }
    }
    pub fn drain(&mut self) -> impl Iterator<Item = (u32, V)> + '_ {
        let items = &mut self.items;
        self.slots[..self.capacity].iter_mut().filter_map(move |e| {
            e.take().map(|(_, k, v)| {
                *items -= 1;
                (k, v)
            })
        })
    }
    pub fn len(&self) -> usize {
        self.items
    }
}

pub enum Entry<'a, K, V> {
    Occupied(&'a mut V),
    Vacant(K, &'a mut Option<(NonZeroU8, K, V)>, &'a mut usize),
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
            Entry::Vacant(k, slot, items) => {
                *slot = Some((NonZeroU8::new(1).unwrap(), k, v));
                *items += 1;
                slot.as_mut().map(|e| &mut e.2).unwrap()
            }
        }
    }
}
