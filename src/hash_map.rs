use std::convert::{TryFrom, TryInto};

pub(crate) struct HashMap<V> {
    slots: Box<[Option<(u32, V)>]>,
    capacity: usize,
}

impl<V> HashMap<V> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.checked_next_power_of_two().unwrap();
        Self {
            slots: std::iter::repeat_with(|| None)
                .take(capacity)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            capacity,
        }
    }
    pub(crate) fn shrink_to(&mut self, capacity: usize) {
        let new_capacity = capacity.checked_next_power_of_two().unwrap();
        assert!(new_capacity <= self.slots.len());
        self.capacity = new_capacity;
    }
    #[inline]
    pub(crate) fn entry(&mut self, key: usize) -> Entry<'_, V> {
        const HASH_SCAL: usize = 107;
        let mut hash = (key * HASH_SCAL) & (self.capacity - 1);
        loop {
            // We redo the borrow in the success cases to avoid a borrowck weakness
            match &self.slots[hash] {
                Some((k, _)) if usize::try_from(*k).unwrap() == key => {
                    break Entry::Occupied(&mut self.slots[hash].as_mut().unwrap().1);
                }
                Some(_) => {
                    hash = (hash + 1) & (self.capacity - 1);
                }
                None => {
                    break Entry::Vacant(key.try_into().unwrap(), &mut self.slots[hash]);
                }
            }
        }
    }
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = (usize, V)> + '_ {
        self.slots[..self.capacity]
            .iter_mut()
            .filter_map(|e| e.take().map(|(i, v)| (i.try_into().unwrap(), v)))
    }
}

pub(crate) enum Entry<'a, V> {
    Occupied(&'a mut V),
    Vacant(u32, &'a mut Option<(u32, V)>),
}

impl<'a, V> Entry<'a, V> {
    #[inline]
    pub(crate) fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(ref mut v) = self {
            f(*v);
        }
        self
    }
    #[inline]
    pub(crate) fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(v) => v,
            Entry::Vacant(k, slot) => {
                *slot = Some((k, default));
                slot.as_mut().map(|(_, v)| v).unwrap()
            }
        }
    }
}
