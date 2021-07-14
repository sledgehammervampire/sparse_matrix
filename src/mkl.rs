use std::{
    convert::TryFrom,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    num::TryFromIntError,
    ops::Mul,
};

use mkl_sys::{
    mkl_sparse_d_create_csr, mkl_sparse_destroy, mkl_sparse_spmm, mkl_sparse_z_create_csr,
    sparse_index_base_t::SPARSE_INDEX_BASE_ZERO, sparse_matrix_t,
    sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE, sparse_status_t::*, MKL_Complex16, MKL_INT,
};
use num_enum::TryFromPrimitive;

pub struct MklCsrMatrix<T, const IS_SORTED: bool> {
    pub(crate) rows: MKL_INT,
    pub(crate) cols: MKL_INT,
    pub(crate) vals: Vec<T>,
    pub(crate) indices: Vec<MKL_INT>,
    pub(crate) offsets: Vec<MKL_INT>,
}

pub struct RustMklSparseMatrix<'a, T, const IS_SORTED: bool> {
    handle: sparse_matrix_t,
    _phantom: PhantomData<&'a T>,
}

impl<T, const IS_SORTED: bool> Drop for RustMklSparseMatrix<'_, T, IS_SORTED> {
    fn drop(&mut self) {
        unsafe {
            mkl_sparse_destroy(self.handle);
        }
    }
}

pub struct CMklSparseMatrix<T, const IS_SORTED: bool> {
    pub(crate) handle: sparse_matrix_t,
    pub(crate) _phantom: PhantomData<*const T>,
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
            _phantom: PhantomData,
        }
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
            _phantom: PhantomData,
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
            _phantom: PhantomData,
        })
    }
}

impl<const IS_SORTED: bool> Mul for &CMklSparseMatrix<MKL_Complex16, IS_SORTED> {
    type Output = Result<CMklSparseMatrix<MKL_Complex16, false>, Error>;

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
            _phantom: PhantomData,
        })
    }
}
