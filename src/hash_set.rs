use std::convert::{TryFrom, TryInto};

pub(crate) struct HashSet {
    slots: Box<[Option<u32>]>,
    capacity: usize,
}

impl HashSet {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.checked_next_power_of_two().unwrap();
        Self {
            slots: vec![None; capacity].into_boxed_slice(),
            capacity,
        }
    }
    // resize to a power of 2 no more than original capacity
    pub(crate) fn shrink_to(&mut self, capacity: usize) {
        let new_capacity = capacity.checked_next_power_of_two().unwrap();
        assert!(new_capacity <= self.slots.len());
        self.capacity = new_capacity;
    }
    #[inline]
    pub(crate) fn insert(&mut self, key: usize) {
        const HASH_SCAL: usize = 107;
        let mut hash = (key * HASH_SCAL) & (self.capacity - 1);
        loop {
            if let Some(k) = self.slots[hash] {
                if usize::try_from(k).unwrap() == key {
                    break;
                } else {
                    hash = (hash + 1) & (self.capacity - 1);
                }
            } else {
                self.slots[hash] = Some(key.try_into().unwrap());
                break;
            }
        }
    }
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = u32> + '_ {
        self.slots[..self.capacity]
            .iter_mut()
            .filter_map(|x| x.take())
    }
}
