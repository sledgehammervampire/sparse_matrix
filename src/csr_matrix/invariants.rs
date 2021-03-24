use num::Num;

use super::CsrMatrix;

pub fn csr_invariants<T: Num>(m: &CsrMatrix<T>) -> bool {
    csr_invariant_1(m)
        && csr_invariant_2(m)
        && csr_invariant_3(m)
        && csr_invariant_4(m)
        && csr_invariant_5(m)
        && csr_invariant_6(m)
        && csr_invariant_7(m)
        && csr_invariant_8(m)
}

fn csr_invariant_1<T>(m: &CsrMatrix<T>) -> bool {
    m.ridx
        .iter()
        .all(|(i, s1)| m.ridx.range(..i).fold(0, |sum, (_, s2)| sum + s2.len) == s1.start)
}

fn csr_invariant_2<T>(m: &CsrMatrix<T>) -> bool {
    m.ridx.values().map(|s| s.len).sum::<usize>() == m.vals.len()
}

fn csr_invariant_3<T>(m: &CsrMatrix<T>) -> bool {
    m.cidx.len() == m.vals.len()
}

fn csr_invariant_4<T>(m: &CsrMatrix<T>) -> bool {
    m.ridx.values().all(|s| s.len > 0)
}

fn csr_invariant_5<T>(m: &CsrMatrix<T>) -> bool {
    fn is_increasing(s: &[usize]) -> bool {
        let mut max = None;
        for i in s {
            if Some(i) > max {
                max = Some(i);
            } else {
                return false;
            }
        }
        true
    }

    m.ridx
        .values()
        .all(|s| is_increasing(&m.cidx[s.start..s.start + s.len]))
}

fn csr_invariant_6<T: Num>(m: &CsrMatrix<T>) -> bool {
    m.vals.iter().all(|t| !t.is_zero())
}

fn csr_invariant_7<T: Num>(m: &CsrMatrix<T>) -> bool {
    m.ridx.keys().all(|r| (0..m.rows).contains(r))
}

fn csr_invariant_8<T: Num>(m: &CsrMatrix<T>) -> bool {
    m.cidx.iter().all(|c| (0..m.cols).contains(c))
}
