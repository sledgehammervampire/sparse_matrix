use super::CsrMatrix;
use crate::ComplexNewtype;
use mkl_sys::{
    mkl_sparse_d_create_csr, mkl_sparse_d_export_csr, mkl_sparse_destroy, mkl_sparse_spmm,
    mkl_sparse_z_create_csr, mkl_sparse_z_export_csr,
    sparse_index_base_t::{self, SPARSE_INDEX_BASE_ZERO},
    sparse_matrix_t,
    sparse_operation_t::SPARSE_OPERATION_NON_TRANSPOSE,
    sparse_status_t::*,
    sparse_status_t::{self, SPARSE_STATUS_SUCCESS},
    MKL_Complex16, MKL_free, MKL_malloc, MKL_INT,
};
use num_enum::TryFromPrimitive;
use std::{
    convert::{TryFrom, TryInto},
    marker::PhantomData,
    mem::{align_of, size_of, ManuallyDrop, MaybeUninit},
    num::{NonZeroUsize, TryFromIntError},
    ops::Mul,
    ptr::NonNull,
};

const MKL_ALIGN: i32 = 128;

// allocate memory with MKL_ALIGN-byte alignment
fn alloc<T>(elems: usize) -> Result<NonNull<T>, AllocationError> {
    let size = size_of::<T>()
        .checked_mul(elems)
        .ok_or(AllocationError::WouldOverflow)?
        .try_into()?;
    if size == 0 {
        Err(AllocationError::ZeroSized)
    } else {
        NonNull::new(unsafe { MKL_malloc(size, MKL_ALIGN) }.cast())
            .ok_or(AllocationError::AllocationFailed)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum AllocationError {
    #[error("conversion from usize to u64 failed")]
    TryFromInt(#[from] TryFromIntError),
    #[error("number of requested bytes cannot be stored in a usize")]
    WouldOverflow,
    #[error("zero-sized allocation")]
    ZeroSized,
    #[error("allocation failed")]
    AllocationFailed,
}

pub trait FromMklScalar: Sized {
    type Output: From<Self>;
}
impl FromMklScalar for f64 {
    type Output = f64;
}
impl FromMklScalar for MKL_Complex16 {
    type Output = ComplexNewtype<f64>;
}

pub trait IntoMklScalar: Sized + sealed::Sealed {
    type Output: From<Self>;
}
impl IntoMklScalar for f64 {
    type Output = f64;
}
impl IntoMklScalar for ComplexNewtype<f64> {
    type Output = MKL_Complex16;
}

mod sealed {
    use crate::ComplexNewtype;

    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for ComplexNewtype<f64> {}
}

pub trait MklScalar: FromMklScalar {
    /// # Safety
    /// https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/inspector-executor-sparse-blas-routines/matrix-manipulation-routines/mkl-sparse-create-csr.html
    #[allow(clippy::too_many_arguments)]
    unsafe fn create_csr(
        m: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: MKL_INT,
        cols: MKL_INT,
        rows_start: *mut MKL_INT,
        rows_end: *mut MKL_INT,
        col_indx: *mut MKL_INT,
        values: *mut Self,
    ) -> sparse_status_t::Type;

    /// # Safety
    /// https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/inspector-executor-sparse-blas-routines/matrix-manipulation-routines/mkl-sparse-export-csr.html
    #[allow(clippy::too_many_arguments)]
    unsafe fn export_csr(
        source: sparse_matrix_t,
        indexing: *mut sparse_index_base_t::Type,
        rows: *mut MKL_INT,
        cols: *mut MKL_INT,
        rows_start: *mut *mut MKL_INT,
        rows_end: *mut *mut MKL_INT,
        col_indx: *mut *mut MKL_INT,
        values: *mut *mut Self,
    ) -> sparse_status_t::Type;
}
impl MklScalar for f64 {
    unsafe fn create_csr(
        m: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: MKL_INT,
        cols: MKL_INT,
        rows_start: *mut MKL_INT,
        rows_end: *mut MKL_INT,
        col_indx: *mut MKL_INT,
        values: *mut Self,
    ) -> sparse_status_t::Type {
        mkl_sparse_d_create_csr(
            m, indexing, rows, cols, rows_start, rows_end, col_indx, values,
        )
    }

    unsafe fn export_csr(
        source: sparse_matrix_t,
        indexing: *mut sparse_index_base_t::Type,
        rows: *mut MKL_INT,
        cols: *mut MKL_INT,
        rows_start: *mut *mut MKL_INT,
        rows_end: *mut *mut MKL_INT,
        col_indx: *mut *mut MKL_INT,
        values: *mut *mut Self,
    ) -> sparse_status_t::Type {
        mkl_sparse_d_export_csr(
            source, indexing, rows, cols, rows_start, rows_end, col_indx, values,
        )
    }
}
impl MklScalar for MKL_Complex16 {
    unsafe fn create_csr(
        m: *mut sparse_matrix_t,
        indexing: sparse_index_base_t::Type,
        rows: MKL_INT,
        cols: MKL_INT,
        rows_start: *mut MKL_INT,
        rows_end: *mut MKL_INT,
        col_indx: *mut MKL_INT,
        values: *mut Self,
    ) -> sparse_status_t::Type {
        mkl_sparse_z_create_csr(
            m, indexing, rows, cols, rows_start, rows_end, col_indx, values,
        )
    }

    unsafe fn export_csr(
        source: sparse_matrix_t,
        indexing: *mut sparse_index_base_t::Type,
        rows: *mut MKL_INT,
        cols: *mut MKL_INT,
        rows_start: *mut *mut MKL_INT,
        rows_end: *mut *mut MKL_INT,
        col_indx: *mut *mut MKL_INT,
        values: *mut *mut Self,
    ) -> sparse_status_t::Type {
        mkl_sparse_z_export_csr(
            source, indexing, rows, cols, rows_start, rows_end, col_indx, values,
        )
    }
}

pub struct MklCsrMatrix<T, const IS_SORTED: bool> {
    rows: MKL_INT,
    cols: MKL_INT,
    vals: NonNull<T>,
    indices: NonNull<MKL_INT>,
    offsets: NonNull<MKL_INT>,
}

impl<T, const IS_SORTED: bool> Drop for MklCsrMatrix<T, IS_SORTED> {
    fn drop(&mut self) {
        unsafe {
            MKL_free(self.vals.as_ptr().cast());
            MKL_free(self.indices.as_ptr().cast());
            MKL_free(self.offsets.as_ptr().cast());
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum FromCsrError {
    #[error("integral type conversion failed")]
    TryFromInt(#[from] TryFromIntError),
    #[error("allocation failed")]
    Allocation(#[from] AllocationError),
}

impl<T, const IS_SORTED: bool> TryFrom<CsrMatrix<T, IS_SORTED>>
    for MklCsrMatrix<T::Output, IS_SORTED>
where
    T: IntoMklScalar,
{
    type Error = FromCsrError;

    fn try_from(m: CsrMatrix<T, IS_SORTED>) -> Result<Self, Self::Error> {
        let rows = m.rows.get().try_into()?;
        let cols = m.cols.get().try_into()?;
        let nnz = m.vals.len();
        let vals = alloc(nnz)?;
        for (i, t) in m.vals.into_iter().enumerate() {
            let ptr: *mut T::Output = vals.as_ptr();
            debug_assert!(i < nnz);
            assert_eq!(
                usize::try_from(MKL_ALIGN).unwrap() % align_of::<T::Output>(),
                0
            );
            unsafe {
                ptr.wrapping_add(i).write(t.into());
            }
        }
        debug_assert!(m.indices.len() == nnz);
        let indices = alloc(nnz)?;
        for (i, c) in m.indices.into_iter().enumerate() {
            let ptr: *mut MKL_INT = indices.as_ptr();
            debug_assert!(i < nnz);
            debug_assert!(usize::try_from(MKL_ALIGN).unwrap() % align_of::<MKL_INT>() == 0);
            unsafe {
                ptr.wrapping_add(i).write(c.try_into()?);
            }
        }
        debug_assert!(m.offsets.len() == m.rows.get().checked_add(1).unwrap());
        let offsets = alloc(m.offsets.len())?;
        for (i, o) in m.offsets.into_iter().enumerate() {
            let ptr: *mut MKL_INT = offsets.as_ptr();
            debug_assert!(i <= m.rows.get());
            debug_assert!(usize::try_from(MKL_ALIGN).unwrap() % align_of::<MKL_INT>() == 0);
            unsafe {
                ptr.wrapping_add(i).write(o.try_into()?);
            }
        }
        Ok(MklCsrMatrix {
            rows,
            cols,
            vals,
            indices,
            offsets,
        })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum FromMklCsrError {
    #[error("integral type conversion failed")]
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

impl<'a, T, const IS_SORTED: bool> TryFrom<&'a mut MklCsrMatrix<T, IS_SORTED>>
    for RustMklSparseMatrix<'a, T, IS_SORTED>
where
    T: MklScalar,
{
    type Error = MklError;

    fn try_from(m: &'a mut MklCsrMatrix<T, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut handle = MaybeUninit::uninit();
        let rows_start = m.offsets.as_ptr();
        debug_assert!(m.rows >= 1);
        // SAFETY: m.offsets points to an allocation of size m.rows+1
        let rows_end = rows_start.wrapping_add(1);
        // SAFETY: rows_start and rows_end point to an allocation of size m.rows+1
        // SAFETY: m.indices and m.vals point to an allocation of size m.offsets[m.rows]
        let status = unsafe {
            T::create_csr(
                handle.as_mut_ptr(),
                SPARSE_INDEX_BASE_ZERO,
                m.rows,
                m.cols,
                rows_start,
                rows_end,
                m.indices.as_ptr(),
                m.vals.as_ptr(),
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

impl<T, const IS_SORTED: bool> TryFrom<CMklSparseMatrix<T, IS_SORTED>>
    for CsrMatrix<T::Output, IS_SORTED>
where
    T: MklScalar,
{
    type Error = FromMklCsrError;

    fn try_from(m: CMklSparseMatrix<T, IS_SORTED>) -> Result<Self, Self::Error> {
        let mut indexing = MaybeUninit::uninit();
        let mut rows = MaybeUninit::uninit();
        let mut cols = MaybeUninit::uninit();
        let mut rows_start = MaybeUninit::uninit();
        let mut rows_end = MaybeUninit::uninit();
        let mut col_indx = MaybeUninit::uninit();
        let mut values = MaybeUninit::uninit();
        let status = unsafe {
            T::export_csr(
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
            return Err(FromMklCsrError::Mkl(MklError::try_from(status).unwrap()));
        }
        let indexing = usize::try_from(unsafe { indexing.assume_init() })?;
        debug_assert!(indexing == SPARSE_INDEX_BASE_ZERO.try_into().unwrap());
        let rows = NonZeroUsize::try_from(usize::try_from(unsafe { rows.assume_init() })?)?;
        let cols = NonZeroUsize::try_from(usize::try_from(unsafe { cols.assume_init() })?)?;
        let rows_start = unsafe { rows_start.assume_init() };
        let rows_end = unsafe { rows_end.assume_init() };
        let nnz = usize::try_from(unsafe { *rows_end.wrapping_add(rows.get() - 1) })?;
        let col_indx = unsafe { col_indx.assume_init() };
        let values = unsafe { values.assume_init() };

        let mut offsets = vec![];
        for i in 0..rows.get() {
            offsets.push(unsafe { rows_start.wrapping_add(i).read() }.try_into()?);
        }
        offsets.push(nnz);
        let mut indices = vec![];
        for i in 0..nnz {
            indices.push(unsafe { col_indx.wrapping_add(i).read() }.try_into()?);
        }
        let mut vals = vec![];
        for i in 0..nnz {
            vals.push(unsafe { values.wrapping_add(i).read() }.into());
        }
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
    type Output = Result<CMklSparseMatrix<T, false>, FromMklCsrError>;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs: CMklSparseMatrix<_, IS_SORTED> = CMklSparseMatrix {
            handle: self.handle,
            phantom: PhantomData,
        };
        let lhs = ManuallyDrop::new(lhs);
        let rhs: CMklSparseMatrix<_, IS_SORTED> = CMklSparseMatrix {
            handle: rhs.handle,
            phantom: PhantomData,
        };
        let rhs = ManuallyDrop::new(rhs);
        &*lhs * &*rhs
    }
}

impl<T, const IS_SORTED: bool> Mul for &CMklSparseMatrix<T, IS_SORTED> {
    type Output = Result<CMklSparseMatrix<T, false>, FromMklCsrError>;

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
            return Err(FromMklCsrError::Mkl(MklError::try_from(status).unwrap()));
        }
        Ok(CMklSparseMatrix {
            handle: unsafe { res.assume_init() },
            phantom: PhantomData,
        })
    }
}
