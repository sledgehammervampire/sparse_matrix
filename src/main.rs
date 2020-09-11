use num::traits::NumAssignRef;
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::AddAssign,
};

fn main() {
    let mut m1 = CSRMatrix::<u64>::identity(2);
    let m2 = CSRMatrix::identity(2);
    m1 += &m2;
    dbg!(DOKMatrix::from(m1.clone()));
    dbg!(m1);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CSRMatrix<T> {
    rows: usize,
    cols: usize,
    // for v in vals, vals != 0
    vals: Vec<T>,
    cidx: Vec<usize>,
    // for s in ridx.values(), s.len > 0
    ridx: BTreeMap<usize, Slice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Slice {
    start: usize,
    len: usize,
}

impl<T: NumAssignRef + Clone> CSRMatrix<T> {
    pub fn new_square(n: usize) -> CSRMatrix<T> {
        CSRMatrix::new(n, n)
    }
    pub fn new(rows: usize, cols: usize) -> CSRMatrix<T> {
        let capacity = 1000.min(rows * cols / 5);
        CSRMatrix {
            rows,
            cols,
            vals: Vec::with_capacity(capacity),
            cidx: Vec::with_capacity(capacity),
            ridx: BTreeMap::new(),
        }
    }
    pub fn identity(n: usize) -> CSRMatrix<T> {
        let mut m = CSRMatrix::new_square(n);
        m.vals = vec![T::one(); n];
        m.cidx = (0..n).map(|i| i).collect();
        m.ridx = (0..n).map(|i| (i, Slice { start: i, len: 1 })).collect();
        m
    }

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn get_row_start(&self, i: usize) -> usize {
        self.ridx
            .iter()
            .skip_while(|(r, _)| **r < i)
            .next()
            .map(|(_, Slice { start, .. })| start)
            .copied()
            .unwrap_or_else(|| self.cidx.len())
    }
    fn get_row_entries(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        let s = self.ridx.get(&i).copied().unwrap_or_else(|| Slice {
            start: self.get_row_start(i),
            len: 0,
        });
        let r = || s.start..s.start + s.len;
        self.cidx[r()].iter().copied().zip(self.vals[r()].iter())
    }
    fn get_element(&self, (i, j): (usize, usize)) -> Cow<T> {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );
        self.get_row_entries(i)
            .find(|&(c, _)| c == j)
            .map_or(Cow::Owned(T::zero()), |(_, t)| Cow::Borrowed(t))
    }
    fn set_element(&mut self, (i, j): (usize, usize), t: T) {
        assert!(
            (..self.rows).contains(&i) && (..self.cols).contains(&j),
            "values are not in bounds"
        );
        if t.is_zero() {
            let start = self.get_row_start(i);
            if let Some(len) = self.ridx.get(&i).map(|s| s.len) {
                if let Ok(k) = self.cidx[start..start + len].binary_search(&j) {
                    self.vals.remove(start + k);
                    self.cidx.remove(start + k);
                    for (_, s) in self.ridx.range_mut(i + 1..) {
                        s.start -= 1;
                    }
                    let s = self.ridx.get_mut(&i).unwrap();
                    s.len -= 1;
                    if s.len == 0 {
                        self.ridx.remove(&i);
                    }
                }
            }
        } else {
            let start = self.get_row_start(i);
            let len = self.ridx.get(&i).map(|s| s.len).unwrap_or(0);
            match self.cidx[start..start + len].binary_search(&j) {
                Ok(k) => {
                    self.vals[start + k] = t;
                }
                Err(k) => {
                    self.vals.insert(start + k, t);
                    self.cidx.insert(start + k, j);
                    let entry = self.ridx.entry(i).or_insert(Slice { start, len: 0 });
                    entry.len += 1;
                    for (_, s) in self.ridx.range_mut(i + 1..) {
                        s.start += 1;
                    }
                }
            }
        }
    }
}

impl<T: NumAssignRef + Clone> AddAssign<&CSRMatrix<T>> for CSRMatrix<T> {
    fn add_assign(&mut self, rhs: &CSRMatrix<T>) {
        assert_eq!(
            (self.rows, self.cols),
            (rhs.rows, rhs.cols),
            "matrices must have identical dimensions"
        );

        let (mut vals, mut cidx, mut ridx, mut start) = (vec![], vec![], BTreeMap::new(), 0);
        let rs = self
            .ridx
            .keys()
            .chain(rhs.ridx.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        for r in rs {
            let mut row = self
                .get_row_entries(r)
                .map(|(c, t)| (c, t.clone()))
                .collect::<BTreeMap<_, _>>();
            for (col, val) in rhs.get_row_entries(r) {
                let entry = row.entry(col).or_insert(T::zero());
                *entry += val;
                if entry.is_zero() {
                    row.remove(&col);
                }
            }
            let len = row.len();
            let (mut ks, mut vs): (Vec<_>, Vec<_>) = row.into_iter().unzip();
            cidx.append(&mut ks);
            vals.append(&mut vs);
            if len > 0 {
                ridx.insert(r, Slice { start, len });
            }
            start += len;
        }
        self.vals = vals;
        self.cidx = cidx;
        self.ridx = ridx;
    }
}

#[derive(Debug, Eq, PartialEq)]
struct DOKMatrix<T> {
    rows: usize,
    cols: usize,
    entries: HashMap<(usize, usize), T>,
}

impl<T: NumAssignRef + Clone> From<DOKMatrix<T>> for CSRMatrix<T> {
    fn from(old: DOKMatrix<T>) -> Self {
        let mut new = CSRMatrix::new(old.rows, old.cols);
        for ((i, j), t) in old.entries {
            new.set_element((i, j), t);
        }
        new
    }
}
impl<T: NumAssignRef + Clone> From<CSRMatrix<T>> for DOKMatrix<T> {
    fn from(old: CSRMatrix<T>) -> Self {
        let mut entries = HashMap::new();
        let mut vals = old.vals.into_iter().map(Some).collect::<Vec<_>>();
        for (r, Slice { start, len }) in old.ridx {
            for (&c, t) in old.cidx[start..start + len]
                .iter()
                .zip(vals[start..start + len].iter_mut())
            {
                entries.insert((r, c), t.take().unwrap());
            }
        }
        DOKMatrix {
            rows: old.rows,
            cols: old.cols,
            entries,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{CSRMatrix, DOKMatrix};

    #[test]
    fn test_1() {
        assert_eq!(
            DOKMatrix::from(CSRMatrix::identity(2)),
            DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect()
            }
        )
    }
    #[test]
    fn test_2() {
        assert_eq!(
            CSRMatrix::from({
                let entries = vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect();
                DOKMatrix {
                    rows: 2,
                    cols: 2,
                    entries,
                }
            }),
            CSRMatrix::identity(2)
        );
    }
    #[test]
    fn test_3() {
        assert_eq!(
            {
                let mut m1 = CSRMatrix::identity(3);
                let m2 = CSRMatrix::identity(3);
                m1 += &m2;
                m1
            },
            CSRMatrix::from(DOKMatrix {
                rows: 3,
                cols: 3,
                entries: vec![((0, 0), 2), ((1, 1), 2), ((2, 2), 2)]
                    .into_iter()
                    .collect()
            })
        )
    }
    #[test]
    fn test_4() {
        assert_eq!(
            {
                let mut m = CSRMatrix::identity(2);
                m.set_element((0, 1), 1);
                m
            },
            CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                    .into_iter()
                    .collect()
            })
        )
    }
    #[test]
    fn test_5() {
        assert_eq!(
            {
                let mut m = CSRMatrix::from(DOKMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                        .into_iter()
                        .collect(),
                });
                m.set_element((0, 1), 0);
                m
            },
            CSRMatrix::identity(2)
        )
    }
    #[test]
    fn test_6() {
        assert_eq!(
            {
                let mut m = CSRMatrix::from(DOKMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![((0, 0), 1), ((0, 1), 1), ((1, 1), 1)]
                        .into_iter()
                        .collect(),
                });
                m.set_element((0, 1), 2);
                m
            },
            CSRMatrix::from(DOKMatrix {
                rows: 2,
                cols: 2,
                entries: vec![((0, 0), 1), ((0, 1), 2), ((1, 1), 1)]
                    .into_iter()
                    .collect(),
            })
        )
    }
    #[test]
    fn test_7() {
        assert_eq!(
            {
                let mut m = CSRMatrix::from(DOKMatrix {
                    rows: 2,
                    cols: 2,
                    entries: vec![((0, 0), 1), ((1, 1), 1)].into_iter().collect(),
                });
                m.set_element((0, 1), 0);
                m
            },
            CSRMatrix::identity(2)
        )
    }
}
