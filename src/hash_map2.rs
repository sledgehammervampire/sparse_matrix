use std::{convert::TryFrom, fmt::Debug, iter::repeat_with, mem::MaybeUninit};

// keys.len() == values.len()
// keys.len() is a power of 2
// keys[hash].is_some() ==> values[hash] initialized
pub(crate) struct HashMap<K, V> {
    keys: Box<[Option<K>]>,
    values: Box<[MaybeUninit<V>]>,
    capacity: usize,
}

impl<K: TryFrom<usize> + Copy> HashMap<K, V>
where
    // no Drop impl, no need to worry about leaks in values
    V: Copy,
    for<'a> &'a K: Eq,
    usize: TryFrom<K>,
    <usize as TryFrom<K>>::Error: Debug,
{
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.checked_next_power_of_two().unwrap();
        Self {
            keys: repeat_with(|| None)
                .take(capacity)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            values: repeat_with(MaybeUninit::uninit)
                .take(capacity)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            capacity,
        }
    }
    pub(crate) fn shrink_to(&mut self, capacity: usize) {
        let new_capacity = capacity.checked_next_power_of_two().unwrap();
        assert!(new_capacity <= self.keys.len());
        self.capacity = new_capacity;
    }
    #[inline]
    pub(crate) fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        const HASH_SCAL: usize = 107;
        let mut hash: usize = (usize::try_from(key).unwrap() * HASH_SCAL) & (self.capacity - 1);
        loop {
            if let Some(k) = &self.keys[hash] {
                if k == &key {
                    // SAFETY: keys[hash].is_some() ==> values[hash] initialized
                    break Entry::Occupied(unsafe { self.values[hash].assume_init_mut() });
                } else {
                    hash = (hash + 1) & (self.capacity - 1);
                }
            } else {
                break Entry::Vacant(key, &mut self.keys[hash], &mut self.values[hash]);
            }
        }
    }
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
        self.keys[..self.capacity]
            .iter_mut()
            .zip(self.values[..self.capacity].iter_mut())
            .filter_map(move |(i, v)| {
                i.take().map(|i| {
                    (
                        i,
                        // SAFETY: keys[hash].is_some() ==> values[hash] initialized
                        unsafe { v.as_ptr().read() },
                    )
                })
            })
    }
}

pub(crate) enum Entry<'a, K, V> {
    Occupied(&'a mut V),
    Vacant(K, &'a mut Option<K>, &'a mut MaybeUninit<V>),
}

impl<'a, K, V> Entry<'a, K, V> {
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
            Entry::Vacant(k, slot, v) => {
                // we have slot == &mut self.keys[hash] for some hash in 0..values.len()
                *slot = Some(k);
                unsafe {
                    // SAFETY: v == &mut self.values[hash]
                    // maintains self.keys[hash].is_some() ==> self.values[hash] initialized
                    v.as_mut_ptr().write(default);
                    // SAFETY: self.values[hash] just initialized
                    v.assume_init_mut()
                }
            }
        }
    }
}
