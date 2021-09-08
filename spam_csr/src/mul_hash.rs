use std::convert::TryInto;

use itertools::Itertools;
use num_traits::NumAssign;
use rayon::prelude::*;
use spam_matrix::Matrix;

use crate::{checked_inclusive_scan, CsrMatrix};

impl<T: NumAssign + Copy + Send + Sync, const B: bool> CsrMatrix<T, B> {
    // based off pengdada/mtspgemmlib
    // requires: rhs column indices be less than u32::MAX (1<<32 - 1)
    pub fn mul_hash<const B1: bool, const B2: bool>(
        &self,
        rhs: &CsrMatrix<T, B1>,
    ) -> CsrMatrix<T, B2> {
        let (mut row_nz, rows_offset) = self.rows_to_threads(rhs);
        Self::mul_hash_symbolic(self, rhs, &mut row_nz, &rows_offset);
        let (indices, vals, offsets) =
            unsafe { Self::mul_hash_numeric::<B1, B2>(self, rhs, &row_nz, &rows_offset) };
        CsrMatrix {
            rows: self.rows,
            cols: rhs.cols,
            indices,
            vals,
            offsets,
        }
    }

    fn rows_to_threads<const B1: bool>(&self, rhs: &CsrMatrix<T, B1>) -> (Vec<usize>, Vec<usize>) {
        let row_nz: Vec<_> = self
            .offsets
            .par_iter()
            .zip(self.offsets.par_iter().skip(1))
            .map(|(&a, &b)| {
                self.indices[a..b]
                    .iter()
                    .map(|&k| rhs.offsets[k + 1] - rhs.offsets[k])
                    .try_fold(0usize, |sum, x| sum.checked_add(x))
                    .unwrap()
            })
            .collect();
        let ps_row_nz = checked_inclusive_scan(&row_nz);
        let total_intprod = ps_row_nz.last().copied().unwrap();
        // TODO: not sure what the equivalent of omp_get_max_threads is
        let tnum = num_cpus::get();
        let average_intprod = total_intprod.div_ceil(tnum);
        let mut rows_offset = vec![0];
        rows_offset.par_extend(
            (1..tnum)
                .into_par_iter()
                .map(|tid| ps_row_nz.partition_point(|&x| x <= average_intprod * tid) - 1),
        );
        rows_offset.push(self.rows.get());
        (row_nz, rows_offset)
    }

    fn mul_hash_symbolic<const B1: bool>(
        self: &CsrMatrix<T, B>,
        rhs: &CsrMatrix<T, B1>,
        mut rest: &mut [usize],
        rows_offset: &[usize],
    ) {
        rayon::scope(move |s| {
            for (tlo, thi) in rows_offset.iter().copied().tuple_windows() {
                let (s1, s2) = rest.split_at_mut(thi - tlo);
                rest = s2;
                s.spawn(move |_| {
                    let max_capacity = s1.iter().copied().max().unwrap_or(0);
                    let mut hs = linprobe::HashSet::with_capacity(max_capacity);
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
                        hs.shrink_to(*row_nz);
                        for &k in &self.indices[row_start..row_end] {
                            let (rlo, rhi) = (rhs.offsets[k], rhs.offsets[k + 1]);
                            for &j in &rhs.indices[rlo..rhi] {
                                // try_into causes perf regression
                                hs.insert(j as u32);
                            }
                        }
                        *row_nz = hs.len();
                        hs.clear();
                    }
                    #[cfg(feature = "debug")]
                    dbg!(&hs.probe_lengths);
                });
            }
        });
    }

    #[allow(unused_unsafe)]
    unsafe fn mul_hash_numeric<const B1: bool, const B2: bool>(
        &self,
        rhs: &CsrMatrix<T, B1>,
        row_nz: &[usize],
        rows_offset: &[usize],
    ) -> (Vec<usize>, Vec<T>, Vec<usize>) {
        debug_assert_eq!(*rows_offset.first().unwrap(), 0);
        debug_assert_eq!(*rows_offset.last().unwrap(), self.rows().get());
        debug_assert!(rows_offset.is_sorted());
        debug_assert_eq!(row_nz.len(), self.rows.get());

        let offsets = checked_inclusive_scan(row_nz);
        let nnz = *offsets.last().unwrap();
        let (mut indices, mut vals) = (Vec::with_capacity(nnz), Vec::with_capacity(nnz));
        rayon::scope(|s| {
            let (mut indices_rest, mut vals_rest) =
                (indices.spare_capacity_mut(), vals.spare_capacity_mut());
            for (tlo, thi) in rows_offset.iter().copied().tuple_windows() {
                let trow_nz = &row_nz[tlo..thi];
                let (tindices, s2) = indices_rest.split_at_mut(offsets[thi] - offsets[tlo]);
                indices_rest = s2;
                let (tvals, s2) = vals_rest.split_at_mut(offsets[thi] - offsets[tlo]);
                vals_rest = s2;
                let offsets = &offsets;
                s.spawn(move |_| {
                    // maximum of per row capacity
                    let capacity = trow_nz
                        .iter()
                        .copied()
                        .max()
                        .unwrap_or(0)
                        .checked_mul(2)
                        .expect("multiplication by 2 overflowed");
                    let mut hm = linprobe::HashMap::with_capacity(capacity);
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
                            .checked_mul(2)
                            .expect("multiplication by 2 overflowed");
                        hm.shrink_to(capacity);
                        for (k, &t) in self.indices[row_start..row_end]
                            .iter()
                            .copied()
                            .zip(&self.vals[row_start..row_end])
                        {
                            let (rcidx, rvals) = {
                                let (rlo, rhi) = (rhs.offsets[k], rhs.offsets[k + 1]);
                                (&rhs.indices[rlo..rhi], &rhs.vals[rlo..rhi])
                            };
                            for (j, t1) in rcidx.iter().copied().zip(rvals.iter().map(|&t1| t * t1))
                            {
                                // try_into causes perf regression
                                hm.entry(j as u32)
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
                                    tindices[curr].write(c.try_into().unwrap());
                                    tvals[curr].write(t);
                                }
                                curr += 1;
                            }
                        } else {
                            for (c, t) in hm.drain() {
                                // SAFETY: tindices of each thread are disjoint slices of indices
                                // SAFETY: tvals of each thread are disjoint slices of vals
                                unsafe {
                                    tindices[curr].write(c.try_into().unwrap());
                                    tvals[curr].write(t);
                                }
                                curr += 1;
                            }
                        }
                    }
                    #[cfg(feature = "debug")]
                    dbg!(&hm.probe_lengths);
                    assert_eq!(curr, offsets[thi] - offsets[tlo]);
                });
            }
        });
        // SAFETY: rows_offsets induces a partition of the allocated capacity of indices and vals
        // given by (rows_start, rows_end)
        unsafe {
            indices.set_len(indices.capacity());
            vals.set_len(vals.capacity());
        }
        (indices, vals, offsets)
    }
}

#[test]
fn test_rows_to_threads() {
    use proptest::{prelude::*, test_runner::TestRunner};
    use spam_matrix::{proptest::arb_mul_pair, MulPair};

    let mut runner = TestRunner::default();
    runner
        .run(
            &arb_mul_pair::<std::num::Wrapping<i8>, _, _>(
                CsrMatrix::<_, false>::arb_fixed_size_matrix,
            ),
            |MulPair(m1, m2)| {
                let (flop, offsets) = m1.rows_to_threads(&m2);
                prop_assert!(flop.len() == m1.rows().get());
                prop_assert!(offsets.is_sorted(), "{:?}", offsets);
                prop_assert!(*offsets.last().unwrap() == m1.rows().get(), "{:?}", offsets);
                Ok(())
            },
        )
        .unwrap();
}
