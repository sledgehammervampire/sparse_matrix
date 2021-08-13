use cap_rand::prelude::*;
use core::slice;
use itertools::{iproduct, Itertools};
use num::{traits::NumAssign, Num};
use rayon::prelude::*;
use std::{
    alloc::{self, Allocator},
    collections::BTreeMap,
    fmt::Debug,
    iter::repeat_with,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    ops::{Add, Mul, Sub},
    vec,
};

use crate::{dok_matrix::DokMatrix, IndexError, Matrix};

#[cfg(feature = "mkl")]
pub mod mkl;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrMatrix<T, const IS_SORTED: bool> {
    rows: NonZeroUsize,
    cols: NonZeroUsize,
    vals: Vec<T>,
    indices: Vec<usize>,
    offsets: Vec<usize>,
}

impl<T: Num, const IS_SORTED: bool> CsrMatrix<T, IS_SORTED> {
    pub fn row_nnz_freq(&self) -> BTreeMap<usize, usize> {
        let mut freq = BTreeMap::new();
        for (a, b) in self.offsets.iter().copied().tuple_windows() {
            *freq.entry(b - a).or_insert(0) += 1;
        }
        freq
    }

    fn get_row_entries(&self, i: usize) -> (&[usize], &[T]) {
        let (j, k) = (self.offsets[i], self.offsets[i + 1]);
        (&self.indices[j..k], &self.vals[j..k])
    }

    pub(crate) fn into_iter(self) -> impl Iterator<Item = ((usize, usize), T)> {
        let offsets = self.offsets;
        self.indices
            .into_iter()
            .zip(self.vals.into_iter())
            .enumerate()
            .map(move |(i, (c, t))| ((offsets.partition_point(|x| *x <= i) - 1, c), t))
    }

    pub fn iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.offsets
            .iter()
            .copied()
            .tuple_windows()
            .enumerate()
            .flat_map(move |(r, (rlo, rhi))| {
                let (indices, vals) = (&self.indices[rlo..rhi], &self.vals[rlo..rhi]);
                indices.iter().map(move |c| (r, *c)).zip(vals.iter())
            })
    }

    fn invariant1(&self) -> bool {
        self.indices.len() == self.vals.len()
    }

    fn invariant2(&self) -> bool {
        self.offsets.len() == self.rows.get().checked_add(1).unwrap()
    }

    fn invariant3(&self) -> bool {
        crate::is_sorted(&self.offsets)
    }

    fn invariant4(&self) -> bool {
        self.offsets[self.rows.get()] == self.indices.len()
    }

    fn invariant5(&self) -> bool {
        self.indices
            .iter()
            .all(|c| (0..self.cols.get()).contains(c))
    }

    fn invariant6(&self) -> bool {
        self.offsets.iter().copied().tuple_windows().all(|(a, b)| {
            if IS_SORTED {
                crate::is_increasing(&self.indices[a..b])
            } else {
                crate::all_distinct(&self.indices[a..b])
            }
        })
    }

    fn invariant7(&self) -> bool {
        self.offsets[0] == 0
    }

    fn apply_elementwise<F>(mut self, mut rhs: Self, f: &F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );
        let (mut vals, mut cidx, mut ridx) = (vec![], vec![], vec![0]);

        for ((a, b), (c, d)) in self
            .offsets
            .iter()
            .copied()
            .tuple_windows()
            .zip(rhs.offsets.iter().copied().tuple_windows())
        {
            let (mut rcidx, mut rvals) = if IS_SORTED {
                self.indices[a..b]
                    .iter()
                    .copied()
                    .zip(self.vals.splice(a..b, repeat_with(T::zero).take(b - a)))
                    .merge_join_by(
                        rhs.indices[c..d]
                            .iter()
                            .copied()
                            .zip(rhs.vals.splice(c..d, repeat_with(T::zero).take(d - c))),
                        |(c1, _), (c2, _)| c1.cmp(c2),
                    )
                    .map(|eob| match eob {
                        itertools::EitherOrBoth::Both((c, t1), (_, t2)) => (c, f(t1, t2)),
                        itertools::EitherOrBoth::Left((c, t)) => (c, f(t, T::zero())),
                        itertools::EitherOrBoth::Right((c, t)) => (c, f(T::zero(), t)),
                    })
                    .unzip()
            } else {
                let mut row: hashbrown::HashMap<_, _> = self.indices[a..b]
                    .iter()
                    .copied()
                    .zip(self.vals.splice(a..b, repeat_with(T::zero).take(b - a)))
                    .collect();
                for (c, t1) in rhs.indices[c..d]
                    .iter()
                    .copied()
                    .zip(rhs.vals.splice(c..d, repeat_with(T::zero).take(d - c)))
                {
                    let entry = row.entry(c).or_insert_with(T::zero);
                    *entry = f(mem::replace(entry, T::zero()), t1);
                }
                row.into_iter().unzip()
            };

            vals.append(&mut rvals);
            cidx.append(&mut rcidx);
            ridx.push(vals.len());
        }

        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            vals,
            indices: cidx,
            offsets: ridx,
        }
    }
}

#[test]
fn test_into_iter() {
    use proptest::prop_assert_eq;

    let config = proptest::test_runner::Config {
        max_shrink_iters: 10000,
        ..Default::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(&CsrMatrix::<i8, false>::arb_matrix(), |m| {
            prop_assert_eq!(
                m.iter().map(|e| (e.0, *e.1)).collect::<Vec<_>>(),
                m.into_iter().collect::<Vec<_>>()
            );
            Ok(())
        })
        .unwrap();
}

impl<T: Num, const IS_SORTED: bool> Matrix<T> for CsrMatrix<T, IS_SORTED> {
    fn invariants(&self) -> bool {
        self.invariant1()
            && self.invariant2()
            && self.invariant3()
            && self.invariant4()
            && self.invariant5()
            && self.invariant6()
            && self.invariant7()
    }

    fn new((rows, cols): (NonZeroUsize, NonZeroUsize)) -> Self {
        let capacity = 1000.min(rows.get() * cols.get() / 5);
        CsrMatrix {
            rows,
            cols,
            vals: Vec::with_capacity(capacity),
            indices: Vec::with_capacity(capacity),
            offsets: vec![0; rows.get() + 1],
        }
    }

    fn new_square(n: NonZeroUsize) -> Self {
        Self::new((n, n))
    }

    fn identity(n: NonZeroUsize) -> Self {
        CsrMatrix {
            rows: n,
            cols: n,
            vals: repeat_with(T::one).take(n.get()).collect(),
            indices: (0..n.get()).collect(),
            offsets: (0..=n.get()).collect(),
        }
    }

    fn rows(&self) -> NonZeroUsize {
        self.rows
    }

    fn cols(&self) -> NonZeroUsize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.indices.len()
    }

    fn get_element(&self, (i, j): (usize, usize)) -> Result<Option<&T>, IndexError> {
        if !(i < self.rows.get() && j < self.cols.get()) {
            return Err(IndexError);
        }

        let (cidx, vals) = self.get_row_entries(i);
        Ok(if IS_SORTED {
            cidx.binary_search(&j).ok().map(|k| &vals[k])
        } else {
            cidx.iter().position(|k| *k == j).map(|k| &vals[k])
        })
    }

    fn set_element(&mut self, (i, j): (usize, usize), t: T) -> Result<Option<T>, IndexError> {
        if !(i < self.rows.get() && j < self.cols.get()) {
            return Err(IndexError);
        }

        if IS_SORTED {
            match self.get_row_entries(i).0.binary_search(&j) {
                Ok(pos) => {
                    let pos = self.offsets[i] + pos;
                    Ok(Some(mem::replace(&mut self.vals[pos], t)))
                }
                Err(pos) => {
                    let pos = self.offsets[i] + pos;
                    self.vals.insert(pos, t);
                    self.indices.insert(pos, j);
                    for m in i + 1..=self.rows.get() {
                        self.offsets[m] += 1;
                    }
                    Ok(None)
                }
            }
        } else {
            match self.get_row_entries(i).0.iter().position(|k| *k == j) {
                Some(pos) => {
                    let pos = self.offsets[i] + pos;
                    Ok(Some(mem::replace(&mut self.vals[pos], t)))
                }
                None => {
                    let pos = self.offsets[i + 1];
                    self.vals.insert(pos, t);
                    self.indices.insert(pos, j);
                    for m in i + 1..=self.rows.get() {
                        self.offsets[m] += 1;
                    }
                    Ok(None)
                }
            }
        }
    }

    fn transpose(mut self) -> Self {
        let mut new = CsrMatrix::new((self.cols, self.rows));
        for (j, i) in iproduct!(0..self.cols.get(), 0..self.rows.get()) {
            if let Some(t) = self.set_element((i, j), T::zero()).unwrap() {
                new.set_element((j, i), t).unwrap();
            }
        }
        new
    }
}

impl<T: Num> From<DokMatrix<T>> for CsrMatrix<T, true> {
    fn from(old: DokMatrix<T>) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let entries: Vec<_> = old.entries.into_iter().collect();
        let (mut vals, mut indices, mut offsets) = (vec![], vec![], vec![]);
        for ((i, j), t) in entries {
            offsets.extend(std::iter::repeat(vals.len()).take(i + 1 - offsets.len()));
            vals.push(t);
            indices.push(j);
        }
        offsets.extend(std::iter::repeat(vals.len()).take(rows.get() + 1 - offsets.len()));
        CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        }
    }
}

impl<T: Num> CsrMatrix<T, false> {
    pub fn from_dok(old: DokMatrix<T>, rng: &mut CapRng) -> Self {
        let (rows, cols) = (old.rows(), old.cols());
        let mut entries: Vec<_> = old.entries.into_iter().collect();
        entries.shuffle(rng);
        entries.sort_unstable_by_key(|((i, _), _)| *i);
        let (mut vals, mut indices, mut offsets) = (vec![], vec![], vec![]);
        for ((i, j), t) in entries {
            offsets.extend(std::iter::repeat(vals.len()).take(i + 1 - offsets.len()));
            vals.push(t);
            indices.push(j);
        }
        offsets.extend(std::iter::repeat(vals.len()).take(rows.get() + 1 - offsets.len()));
        CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        }
    }
}

impl<T: NumAssign + Copy + Send + Sync, const B: bool> CsrMatrix<T, B> {
    fn rows_to_threads<const B1: bool>(&self, rhs: &CsrMatrix<T, B1>) -> (Vec<usize>, Vec<usize>) {
        let row_nz: Vec<usize> = self
            .offsets
            .par_iter()
            .zip(self.offsets.par_iter().skip(1))
            .map(|(&a, &b)| {
                self.indices[a..b]
                    .iter()
                    .map(|&k| rhs.get_row_entries(k).0.len())
                    .sum()
            })
            .collect();
        // TODO: make prefix sum parallel
        let ps_row_nz: Vec<_> = std::iter::once(0)
            .chain(row_nz.iter().copied().scan(0, |sum, x| {
                *sum += x;
                Some(*sum)
            }))
            .collect();
        let total_intprod = ps_row_nz.last().copied().unwrap();
        // TODO: not sure what the equivalent of omp_get_max_threads is
        let tnum = num_cpus::get();
        let average_intprod = total_intprod / tnum;
        let mut rows_offset = vec![0];
        rows_offset.par_extend(
            (1..tnum)
                .into_par_iter()
                .map(|tid| ps_row_nz.partition_point(|&x| x < average_intprod * tid)),
        );
        rows_offset.push(self.rows.get());
        (row_nz, rows_offset)
    }

    // based off pengdada/mtspgemmlib
    pub fn mul_hash<const B1: bool, const B2: bool>(
        &self,
        rhs: &CsrMatrix<T, B1>,
    ) -> CsrMatrix<T, B2> {
        let (mut row_nz, rows_offset) = self.rows_to_threads(rhs);
        // mutate row_nz or alloc new vec for new row_nz?
        crossbeam::scope(|s| {
            let mut rest = &mut row_nz[..];
            for (tlo, thi) in rows_offset.iter().copied().tuple_windows() {
                let (s1, s2) = rest.split_at_mut(thi - tlo);
                rest = s2;
                s.spawn(move |_| {
                    // maximum of per row capacity
                    let capacity = s1
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0)
                        .checked_next_power_of_two()
                        .expect("next power of 2 doesn't fit a usize");
                    let mut hs = HashMap::with_capacity(capacity);
                    for ((row_start, row_end), row_nz) in self.offsets[tlo..=thi]
                        .iter()
                        .copied()
                        .tuple_windows()
                        .zip(&mut *s1)
                    {
                        if *row_nz == 0 {
                            continue;
                        }
                        // the number of distinct keys inserted is likely much
                        // less than row_nz without multiplying by 2, so we
                        // don't multiply by 2 here, which would make the
                        // benchmarks slower
                        let capacity = row_nz
                            .checked_next_power_of_two()
                            .expect("next power of 2 doesn't fit a usize");
                        hs.shrink_to(capacity);
                        for &k in &self.indices[row_start..row_end] {
                            for &j in rhs.get_row_entries(k).0 {
                                hs.entry(j).or_insert(());
                            }
                        }
                        *row_nz = hs.drain().count();
                    }
                });
            }
        })
        .unwrap();
        let offsets: Vec<_> = std::iter::once(0)
            .chain(row_nz.iter().copied().scan(0, |sum, x| {
                *sum += x;
                Some(*sum)
            }))
            .collect();
        let nnz = *offsets.last().unwrap();
        let (mut indices, mut vals) = (Vec::with_capacity(nnz), Vec::with_capacity(nnz));
        crossbeam::scope(|s| {
            // SAFETY: indices_rest is the allocated capacity of indices
            // SAFETY: vals_rest is the allocated capacity of vals
            let (mut indices_rest, mut vals_rest) = unsafe {
                (
                    slice::from_raw_parts_mut(
                        indices.as_mut_ptr() as *mut MaybeUninit<usize>,
                        indices.capacity(),
                    ),
                    slice::from_raw_parts_mut(
                        vals.as_mut_ptr() as *mut MaybeUninit<T>,
                        vals.capacity(),
                    ),
                )
            };
            for (tlo, thi) in rows_offset.iter().copied().tuple_windows() {
                let trow_nz = &row_nz[tlo..thi];
                let (tindices, s2) = indices_rest.split_at_mut(offsets[thi] - offsets[tlo]);
                indices_rest = s2;
                let (tvals, s2) = vals_rest.split_at_mut(offsets[thi] - offsets[tlo]);
                vals_rest = s2;
                s.spawn(move |_| {
                    // maximum of per row capacity
                    let capacity = trow_nz
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0)
                        .checked_next_power_of_two()
                        .expect("next power of 2 doesn't fit a usize")
                        .checked_shl(1)
                        .expect("next power of 2 doesn't fit a usize");
                    let mut hm = HashMap::with_capacity(capacity);
                    let mut curr = 0;
                    for ((row_start, row_end), row_nz) in self.offsets[tlo..=thi]
                        .iter()
                        .copied()
                        .tuple_windows()
                        .zip(trow_nz)
                    {
                        if *row_nz == 0 {
                            continue;
                        }
                        // row_nz could be close to row_nz.next_power_of_two(),
                        // so we multiply by 2 to lower the load factor. this
                        // makes benchmarks faster
                        let capacity = row_nz
                            .checked_next_power_of_two()
                            .expect("next power of 2 doesn't fit a usize")
                            .checked_shl(1)
                            .expect("next power of 2 doesn't fit a usize");
                        hm.shrink_to(capacity);
                        for (k, &t) in self.indices[row_start..row_end]
                            .iter()
                            .copied()
                            .zip(&self.vals[row_start..row_end])
                        {
                            let (rcidx, rvals) = rhs.get_row_entries(k);
                            for (j, t1) in rcidx.iter().copied().zip(rvals.iter().map(|&t1| t * t1))
                            {
                                hm.entry(j)
                                    .and_modify(|t| {
                                        *t += t1;
                                    })
                                    .or_insert(t1);
                            }
                        }
                        if B2 {
                            let mut row: Vec<_> = hm.drain().collect();
                            row.sort_unstable_by_key(|(c, _)| *c);
                            for (c, t) in row {
                                // SAFETY: tindices of each thread are disjoint slices of indices
                                // SAFETY: tvals of each thread are disjoint slices of vals
                                unsafe {
                                    tindices[curr].as_mut_ptr().write(c);
                                    tvals[curr].as_mut_ptr().write(t);
                                }
                                curr += 1;
                            }
                        } else {
                            for (c, t) in hm.drain() {
                                // SAFETY: tindices of each thread are disjoint slices of indices
                                // SAFETY: tvals of each thread are disjoint slices of vals
                                unsafe {
                                    tindices[curr].as_mut_ptr().write(c);
                                    tvals[curr].as_mut_ptr().write(t);
                                }
                                curr += 1;
                            }
                        }
                    }
                });
            }
        })
        .unwrap();
        // SAFETY: rows_offsets induces a partition of the allocated capacity of indices and vals
        // given by (rows_start, rows_end)
        unsafe {
            indices.set_len(indices.capacity());
            vals.set_len(vals.capacity());
        }
        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            indices,
            vals,
            offsets,
        }
    }
}

#[cfg(test)]
#[test]
fn test_rows_to_threads() {
    use proptest::{prelude::*, test_runner::TestRunner};
    let mut runner = TestRunner::default();
    runner
        .run(
            &crate::proptest::arb_mul_pair::<std::num::Wrapping<i8>, _, _>(
                CsrMatrix::<_, false>::arb_fixed_size_matrix,
            ),
            |crate::MulPair(m1, m2)| {
                let (flop, offsets) = m1.rows_to_threads(&m2);
                prop_assert!(flop.len() == m1.rows().get());
                prop_assert!(crate::is_sorted(&offsets), "{:?}", offsets);
                prop_assert!(*offsets.last().unwrap() == m1.rows().get(), "{:?}", offsets);
                Ok(())
            },
        )
        .unwrap();
}

impl<T: Num, const IS_SORTED: bool> Add for CsrMatrix<T, IS_SORTED> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::add)
    }
}

impl<T: Num, const IS_SORTED: bool> Sub for CsrMatrix<T, IS_SORTED> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.apply_elementwise(rhs, &T::sub)
    }
}

impl<T: NumAssign + Copy + Send + Sync, const B: bool> Mul for &CsrMatrix<T, B> {
    type Output = CsrMatrix<T, false>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_hash(rhs)
    }
}

#[cfg(test)]
impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T, true> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: NonZeroUsize,
        cols: NonZeroUsize,
    ) -> impl proptest::strategy::Strategy<Value = Self> {
        use proptest::prelude::*;
        repeat_with(|| {
            proptest::sample::subsequence((0..cols.get()).collect::<Vec<_>>(), 0..=cols.get())
        })
        .take(rows.get())
        .collect::<Vec<_>>()
        .prop_flat_map(move |cidx| {
            let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
            for mut rcidx in cidx {
                ridx.push(ridx.last().unwrap() + rcidx.len());
                cidx_flattened.append(&mut rcidx);
            }
            repeat_with(T::arbitrary)
                .take(cidx_flattened.len())
                .collect::<Vec<_>>()
                .prop_map(move |vals| CsrMatrix {
                    rows,
                    cols,
                    vals,
                    indices: cidx_flattened.clone(),
                    offsets: ridx.clone(),
                })
        })
    }

    pub fn arb_matrix() -> impl proptest::strategy::Strategy<Value = Self> {
        crate::proptest::arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

#[cfg(test)]
impl<T: proptest::arbitrary::Arbitrary + Num> CsrMatrix<T, false> {
    pub(crate) fn arb_fixed_size_matrix(
        rows: NonZeroUsize,
        cols: NonZeroUsize,
    ) -> impl proptest::strategy::Strategy<Value = Self> {
        use proptest::prelude::*;
        repeat_with(|| proptest::collection::hash_set(0..cols.get(), 0..=cols.get()))
            .take(rows.get())
            .collect::<Vec<_>>()
            .prop_flat_map(move |cidx| {
                let (mut cidx_flattened, mut ridx) = (vec![], vec![0]);
                for rcidx in cidx {
                    ridx.push(ridx.last().unwrap() + rcidx.len());
                    cidx_flattened.extend(rcidx);
                }
                repeat_with(T::arbitrary)
                    .take(cidx_flattened.len())
                    .collect::<Vec<_>>()
                    .prop_map(move |vals| CsrMatrix {
                        rows,
                        cols,
                        vals,
                        indices: cidx_flattened.clone(),
                        offsets: ridx.clone(),
                    })
            })
    }

    pub fn arb_matrix() -> impl proptest::strategy::Strategy<Value = Self> {
        crate::proptest::arb_matrix::<T, _, _>(Self::arb_fixed_size_matrix)
    }
}

struct HashMap<K, V, A = alloc::Global>
where
    A: Allocator,
{
    slots: Vec<Option<(K, V)>, A>,
    capacity: usize,
}

impl<V: Copy> HashMap<usize, V> {
    fn with_capacity(capacity: usize) -> Self {
        HashMap::with_capacity_in(capacity, alloc::Global)
    }
}
impl<V: Copy, A: Allocator> HashMap<usize, V, A> {
    fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let mut slots = Vec::with_capacity_in(capacity, alloc);
        slots.resize(capacity, None);
        Self { slots, capacity }
    }
    fn shrink_to(&mut self, capacity: usize) {
        debug_assert!(capacity.is_power_of_two());
        debug_assert!(capacity <= self.slots.len());
        self.capacity = capacity;
    }
    fn entry(&mut self, key: usize) -> Entry<'_, usize, V> {
        const HASH_SCAL: usize = 107;
        let mut hash = (key * HASH_SCAL) & (self.capacity - 1);
        loop {
            // We redo the borrow in the success cases to avoid a borrowck weakness
            // TODO: rewrite without reborrow when polonius arrives
            match &self.slots[hash] {
                Some((k, _)) if *k == key => {
                    break Entry::Occupied(&mut self.slots[hash].as_mut().unwrap().1);
                }
                Some(_) => {
                    hash = (hash + 1) & (self.capacity - 1);
                }
                None => {
                    break Entry::Vacant(key, &mut self.slots[hash]);
                }
            }
        }
    }
    fn drain(&mut self) -> impl Iterator<Item = (usize, V)> + '_ {
        self.slots[..self.capacity]
            .iter_mut()
            .filter_map(|e| e.take().map(|(i, v)| (i, v)))
    }
}

enum Entry<'a, K, V> {
    Occupied(&'a mut V),
    Vacant(K, &'a mut Option<(K, V)>),
}

impl<'a, K, V> Entry<'a, K, V> {
    fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        if let Entry::Occupied(ref mut v) = self {
            f(*v);
        }
        self
    }
    fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(v) => v,
            Entry::Vacant(k, slot) => {
                *slot = Some((k, default));
                slot.as_mut().map(|(_, v)| v).unwrap()
            }
        }
    }
}
