use num_enum::TryFromPrimitive;
use std::{
    convert::{TryFrom, TryInto},
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    num::{NonZeroUsize, TryFromIntError},
    ops::Mul,
    slice,
};

use mkl_sys::{
    mkl_sparse_d_create_csr, mkl_sparse_d_export_csr, mkl_sparse_destroy, mkl_sparse_spmm,
    mkl_sparse_z_create_csr, mkl_sparse_z_export_csr, sparse_index_base_t::SPARSE_INDEX_BASE_ZERO,
    sparse_matrix_t, sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE,
    sparse_status_t::SPARSE_STATUS_SUCCESS, sparse_status_t::*, MKL_Complex16, MKL_INT,
};

use crate::ComplexNewtype;

use super::CsrMatrix;

pub struct MklCsrMatrix<T, const IS_SORTED: bool> {
    rows: MKL_INT,
    cols: MKL_INT,
    vals: Vec<T>,
    indices: Vec<MKL_INT>,
    offsets: Vec<MKL_INT>,
}

impl<const IS_SORTED: bool> TryFrom<CsrMatrix<f64, IS_SORTED>> for MklCsrMatrix<f64, IS_SORTED> {
    type Error = TryFromIntError;

    fn try_from(m: CsrMatrix<f64, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = m.rows.get().try_into()?;
        let cols = m.cols.get().try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        Ok(MklCsrMatrix {
            rows,
            cols,
            vals: m.vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<CsrMatrix<ComplexNewtype<f64>, IS_SORTED>>
    for MklCsrMatrix<MKL_Complex16, IS_SORTED>
{
    type Error = TryFromIntError;

    fn try_from(m: CsrMatrix<ComplexNewtype<f64>, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = m.rows.get().try_into()?;
        let cols = m.cols.get().try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(MKL_INT::try_from)
            .collect::<Result<_, _>>()?;
        let vals = m.vals.into_iter().map(MKL_Complex16::from).collect();
        Ok(MklCsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<MklCsrMatrix<f64, IS_SORTED>> for CsrMatrix<f64, IS_SORTED> {
    type Error = TryFromIntError;

    fn try_from(m: MklCsrMatrix<f64, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = usize::try_from(m.rows)?.try_into()?;
        let cols = usize::try_from(m.cols)?.try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        Ok(CsrMatrix {
            rows,
            cols,
            vals: m.vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<MklCsrMatrix<MKL_Complex16, IS_SORTED>>
    for CsrMatrix<ComplexNewtype<f64>, IS_SORTED>
{
    type Error = TryFromIntError;

    fn try_from(m: MklCsrMatrix<MKL_Complex16, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = usize::try_from(m.rows)?.try_into()?;
        let cols = usize::try_from(m.cols)?.try_into()?;
        let offsets: Vec<_> = m
            .offsets
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        let indices: Vec<_> = m
            .indices
            .into_iter()
            .map(usize::try_from)
            .collect::<Result<_, _>>()?;
        let vals = m.vals.into_iter().map(ComplexNewtype::from).collect();
        Ok(CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("out of range integral type conversion attempted")]
    FromInt(#[from] TryFromIntError),
    #[error("mkl error: {0:?}")]
    Mkl(#[from] MklError),
}

#[derive(TryFromPrimitive, thiserror::Error, Debug)]
#[repr(u32)]
pub enum MklError {
    #[error("The routine encountered an empty handle or matrix array.")]
    NotInitialized = SPARSE_STATUS_NOT_INITIALIZED,
    #[error("Internal memory allocation failed.")]
    AllocFailed = SPARSE_STATUS_ALLOC_FAILED,
    #[error("The input parameters contain an invalid value.")]
    InvalidValue = SPARSE_STATUS_INVALID_VALUE,
    #[error("Execution failed.")]
    ExecutionFailed = SPARSE_STATUS_EXECUTION_FAILED,
    #[error("An error in algorithm implementation occurred.")]
    InternalError = SPARSE_STATUS_INTERNAL_ERROR,
    #[error("The requested operation is not supported.")]
    NotSupported = SPARSE_STATUS_NOT_SUPPORTED,
}

pub struct RustMklSparseMatrix<'a, T, const IS_SORTED: bool> {
    handle: sparse_matrix_t,
    phantom: PhantomData<&'a mut T>,
}

impl<T, const IS_SORTED: bool> Drop for RustMklSparseMatrix<'_, T, IS_SORTED> {
    fn drop(&mut self) {
        unsafe {
            mkl_sparse_destroy(self.handle);
        }
    }
}

pub struct CMklSparseMatrix<T, const IS_SORTED: bool> {
    handle: sparse_matrix_t,
    phantom: PhantomData<*mut T>,
}

impl<T, const IS_SORTED: bool> Drop for CMklSparseMatrix<T, IS_SORTED> {
    fn drop(&mut self) {
        unsafe {
            mkl_sparse_destroy(self.handle);
        }
    }
}

impl<T, const IS_SORTED: bool> From<RustMklSparseMatrix<'_, T, IS_SORTED>>
    for CMklSparseMatrix<T, IS_SORTED>
{
    fn from(m: RustMklSparseMatrix<'_, T, IS_SORTED>) -> Self {
        let m = ManuallyDrop::new(m);
        CMklSparseMatrix {
            handle: m.handle,
            phantom: PhantomData,
        }
    }
}

impl<'a, const IS_SORTED: bool> TryFrom<&'a mut MklCsrMatrix<f64, IS_SORTED>>
    for RustMklSparseMatrix<'a, f64, IS_SORTED>
{
    type Error = MklError;

    fn try_from(m: &'a mut MklCsrMatrix<f64, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut handle = MaybeUninit::uninit();
        let rows_start = m.offsets.as_mut_ptr();
        let rows_end = rows_start.wrapping_add(1);
        let status = unsafe {
            mkl_sparse_d_create_csr(
                handle.as_mut_ptr(),
                SPARSE_INDEX_BASE_ZERO,
                m.rows,
                m.cols,
                rows_start,
                rows_end,
                m.indices.as_mut_ptr(),
                m.vals.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(MklError::try_from(status).unwrap());
        }
        Ok(RustMklSparseMatrix {
            handle: unsafe { handle.assume_init() },
            phantom: PhantomData,
        })
    }
}

impl<'a, const IS_SORTED: bool> TryFrom<&'a mut MklCsrMatrix<MKL_Complex16, IS_SORTED>>
    for RustMklSparseMatrix<'a, MKL_Complex16, IS_SORTED>
{
    type Error = MklError;

    fn try_from(m: &'a mut MklCsrMatrix<MKL_Complex16, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut handle = MaybeUninit::uninit();
        let rows_start = m.offsets.as_mut_ptr();
        let rows_end = rows_start.wrapping_add(1);
        let status = unsafe {
            mkl_sparse_z_create_csr(
                handle.as_mut_ptr(),
                SPARSE_INDEX_BASE_ZERO,
                m.rows,
                m.cols,
                rows_start,
                rows_end,
                m.indices.as_mut_ptr(),
                m.vals.as_mut_ptr().cast(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(MklError::try_from(status).unwrap());
        }
        Ok(RustMklSparseMatrix {
            handle: unsafe { handle.assume_init() },
            phantom: PhantomData,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<CMklSparseMatrix<f64, IS_SORTED>>
    for CsrMatrix<f64, IS_SORTED>
{
    type Error = Error;

    fn try_from(m: CMklSparseMatrix<f64, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut indexing = MaybeUninit::uninit();
        let mut rows = MaybeUninit::uninit();
        let mut cols = MaybeUninit::uninit();
        let mut rows_start = MaybeUninit::uninit();
        let mut rows_end = MaybeUninit::uninit();
        let mut col_indx = MaybeUninit::uninit();
        let mut values = MaybeUninit::uninit();
        let status = unsafe {
            mkl_sparse_d_export_csr(
                m.handle,
                indexing.as_mut_ptr(),
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                rows_start.as_mut_ptr(),
                rows_end.as_mut_ptr(),
                col_indx.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(Error::Mkl(MklError::try_from(status).unwrap()));
        }
        let indexing = usize::try_from(unsafe { indexing.assume_init() })?;
        let rows = NonZeroUsize::try_from(usize::try_from(unsafe { rows.assume_init() })?)?;
        let cols = NonZeroUsize::try_from(usize::try_from(unsafe { cols.assume_init() })?)?;
        let mut offsets: Vec<_> =
            unsafe { slice::from_raw_parts(rows_start.assume_init(), rows.get()) }
                .iter()
                .map(|i| Ok(usize::try_from(*i)? - indexing))
                .collect::<Result<_, TryFromIntError>>()?;
        let nnz = unsafe { *rows_end.assume_init().wrapping_add(rows.get() - 1) }.try_into()?;
        offsets.push(nnz);
        let indices: Vec<_> = unsafe { slice::from_raw_parts(col_indx.assume_init(), nnz) }
            .iter()
            .map(|i| Ok(usize::try_from(*i)? - indexing))
            .collect::<Result<_, TryFromIntError>>()?;
        let vals = unsafe { slice::from_raw_parts(values.assume_init(), nnz) }.to_vec();
        Ok(CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

impl<const IS_SORTED: bool> TryFrom<CMklSparseMatrix<MKL_Complex16, IS_SORTED>>
    for CsrMatrix<ComplexNewtype<f64>, IS_SORTED>
{
    type Error = Error;

    fn try_from(m: CMklSparseMatrix<MKL_Complex16, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut indexing = MaybeUninit::uninit();
        let mut rows = MaybeUninit::uninit();
        let mut cols = MaybeUninit::uninit();
        let mut rows_start = MaybeUninit::uninit();
        let mut rows_end = MaybeUninit::uninit();
        let mut col_indx = MaybeUninit::uninit();
        let mut values = MaybeUninit::uninit();
        let status = unsafe {
            mkl_sparse_z_export_csr(
                m.handle,
                indexing.as_mut_ptr(),
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                rows_start.as_mut_ptr(),
                rows_end.as_mut_ptr(),
                col_indx.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(Error::Mkl(MklError::try_from(status).unwrap()));
        }
        let indexing = usize::try_from(unsafe { indexing.assume_init() })?;
        let rows = NonZeroUsize::try_from(usize::try_from(unsafe { rows.assume_init() })?)?;
        let cols = NonZeroUsize::try_from(usize::try_from(unsafe { cols.assume_init() })?)?;
        let mut offsets: Vec<_> =
            unsafe { slice::from_raw_parts(rows_start.assume_init(), rows.get()) }
                .iter()
                .map(|i| Ok(usize::try_from(*i)? - indexing))
                .collect::<Result<_, TryFromIntError>>()?;
        let nnz = unsafe { *rows_end.assume_init().wrapping_add(rows.get() - 1) }.try_into()?;
        offsets.push(nnz);
        let indices: Vec<_> = unsafe { slice::from_raw_parts(col_indx.assume_init(), nnz) }
            .iter()
            .map(|i| Ok(usize::try_from(*i)? - indexing))
            .collect::<Result<_, TryFromIntError>>()?;
        let vals = unsafe { slice::from_raw_parts(values.assume_init().cast(), nnz) }.to_vec();
        Ok(CsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

impl<T, const IS_SORTED: bool> Mul for &RustMklSparseMatrix<'_, T, IS_SORTED> {
    type Output = Result<CMklSparseMatrix<T, false>, Error>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = MaybeUninit::uninit();
        let status = unsafe {
            mkl_sparse_spmm(
                SPARSE_OPERATION_NON_TRANSPOSE,
                self.handle,
                rhs.handle,
                res.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(Error::Mkl(MklError::try_from(status).unwrap()));
        }
        Ok(CMklSparseMatrix {
            handle: unsafe { res.assume_init() },
            phantom: PhantomData,
        })
    }
}

impl<T, const IS_SORTED: bool> Mul for &CMklSparseMatrix<T, IS_SORTED> {
    type Output = Result<CMklSparseMatrix<T, false>, Error>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = MaybeUninit::uninit();
        let status = unsafe {
            mkl_sparse_spmm(
                SPARSE_OPERATION_NON_TRANSPOSE,
                self.handle,
                rhs.handle,
                res.as_mut_ptr(),
            )
        };
        if status != SPARSE_STATUS_SUCCESS {
            return Err(Error::Mkl(MklError::try_from(status).unwrap()));
        }
        Ok(CMklSparseMatrix {
            handle: unsafe { res.assume_init() },
            phantom: PhantomData,
        })
    }
}
