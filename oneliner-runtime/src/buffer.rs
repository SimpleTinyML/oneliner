//! Buffer helpers used by generated execution code.

use super::Error;

/// Intended access to a byte range, as described by generated commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read-only access.
    Ro,
    /// Write-only access.
    Wo,
    /// Read/write access.
    Rw,
    /// Unclassified access; rejected by checked range construction.
    Unknown,
}

/// Non-owning read-only byte descriptor.
#[derive(Clone, Copy)]
pub struct Buffer {
    /// Base address; the descriptor does not keep the allocation alive.
    pub ptr: *const u8,
    /// Declared capacity in bytes.
    pub len: usize,
}

impl Buffer {
    /// Creates a buffer descriptor without dereferencing or borrowing the pointer.
    ///
    /// # Panics
    ///
    /// Panics on a null pointer, including when `len` is zero.
    pub fn new(ptr: *const u8, len: usize) -> Self {
        Self::try_new(ptr, len).expect("invalid buffer reference")
    }

    fn try_new(ptr: *const u8, len: usize) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::NullPointer);
        }
        Ok(Self { ptr, len })
    }
}

/// Non-owning mutable byte descriptor.
#[derive(Clone, Copy)]
pub struct BufferMut {
    /// Base address; writable access must be justified by the consumer.
    pub ptr: *mut u8,
    /// Declared capacity in bytes.
    pub len: usize,
}

impl BufferMut {
    /// Creates a mutable descriptor without validating its allocation.
    ///
    /// # Panics
    ///
    /// Panics on a null pointer, including when `len` is zero.
    pub fn new(ptr: *mut u8, len: usize) -> Self {
        Self::try_new(ptr, len).expect("invalid buffer reference")
    }

    fn try_new(ptr: *mut u8, len: usize) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::NullPointer);
        }
        Ok(Self { ptr, len })
    }
}
/// Converts a generated storage item into the `BufferRef` used by dispatch.
pub trait BufferSource {
    /// Convert a type to inmutable Buffer.
    ///
    /// # Safety
    ///
    /// Keep the storage alive and at the same address until all descriptor
    /// consumers complete. Preserve read access and exclude conflicting writes.
    unsafe fn to_buffer_ref(&self) -> Buffer;

    /// Convert a type to mutable Buffer.
    ///
    /// # Safety
    ///
    /// Keep the storage alive and stationary, and preserve exclusive access until
    /// all consumers complete. The descriptor does not enforce these conditions.
    unsafe fn to_buffer_mut(&mut self) -> BufferMut;
}

impl<const N: usize> BufferSource for [u8; N] {
    unsafe fn to_buffer_ref(&self) -> Buffer {
        Buffer {
            ptr: self as *const u8,
            len: N,
        }
    }
    unsafe fn to_buffer_mut(&mut self) -> BufferMut {
        BufferMut {
            ptr: self as *mut u8,
            len: N,
        }
    }
}

/// Byte subrange plus intended access for a buffer descriptor.
///
/// Public fields allow unchecked construction. Unsafe consumers must validate
/// the actual range even when this value did not come from a checked constructor.
#[derive(Clone, Copy)]
pub struct BufferRange<T> {
    /// Underlying descriptor.
    pub buffer: T,
    /// Intended access, without a lifetime or aliasing guarantee.
    pub access: Access,
    /// Start relative to the base address, in bytes.
    pub offset: usize,
    /// Subrange length in bytes.
    pub length: usize,
}

/// Either a read-only or mutable non-owning descriptor.
#[derive(Clone, Copy)]
pub enum AnyBuffer {
    /// Read-only descriptor.
    Ref(Buffer),
    /// Mutable descriptor.
    Mut(BufferMut),
}

impl AnyBuffer {
    /// Returns the underlying descriptor's declared byte capacity.
    pub fn len(&self) -> usize {
        match self {
            AnyBuffer::Ref(t) => t.len,
            AnyBuffer::Mut(t) => t.len,
        }
    }
}

impl From<Buffer> for AnyBuffer {
    fn from(t: Buffer) -> Self {
        AnyBuffer::Ref(t)
    }
}

impl From<BufferMut> for AnyBuffer {
    fn from(t: BufferMut) -> Self {
        AnyBuffer::Mut(t)
    }
}

/// Range over either descriptor kind.
pub type AnyBufferRange = BufferRange<AnyBuffer>;

impl AnyBufferRange {
    /// Creates a range after numeric bounds and access-kind checks.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_new`] would return an error.
    pub fn new(buffer: AnyBuffer, access: Access, offset: usize, length: usize) -> Self {
        Self::try_new(buffer, access, offset, length).expect("invalid buffer range")
    }

    /// Checks byte bounds and descriptor/access compatibility.
    ///
    /// Read-only descriptors accept `Ro`; mutable descriptors accept `Ro`, `Wo`
    /// or `Rw`. A zero-length range at the declared capacity is allowed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferRangeOutOfBounds`] on overflow or an oversized range,
    /// or [`Error::InvalidAccess`] for incompatible access. Pointer validity and
    /// alignment are not checked.
    ///
    /// # Examples
    ///
    /// ```
    /// use oneliner_runtime::{Access, AnyBufferRange, Buffer, Error};
    /// let bytes = [1_u8, 2, 3, 4];
    /// let buffer = Buffer::new(bytes.as_ptr(), bytes.len());
    /// let range = AnyBufferRange::try_new(buffer.into(), Access::Ro, 1, 2).unwrap();
    /// assert_eq!(range.length, 2);
    /// assert!(matches!(
    ///     AnyBufferRange::try_new(buffer.into(), Access::Ro, 3, 2),
    ///     Err(Error::BufferRangeOutOfBounds { .. })
    /// ));
    /// ```
    pub fn try_new(
        buffer: AnyBuffer,
        access: Access,
        offset: usize,
        length: usize,
    ) -> Result<Self, Error> {
        let buffer_len = buffer.len();

        let end = offset
            .checked_add(length)
            .ok_or(Error::BufferRangeOutOfBounds {
                offset,
                length,
                capacity: buffer_len,
            })?;

        if end > buffer_len {
            return Err(Error::BufferRangeOutOfBounds {
                offset,
                length,
                capacity: buffer_len,
            });
        }

        let access_valid = matches!(
            (buffer, access),
            (AnyBuffer::Ref(_), Access::Ro)
                | (AnyBuffer::Mut(_), Access::Ro)
                | (AnyBuffer::Mut(_), Access::Wo)
                | (AnyBuffer::Mut(_), Access::Rw)
        );

        if !access_valid {
            return Err(Error::InvalidAccess {
                access,
                required: match buffer {
                    AnyBuffer::Ref(_) => Access::Ro,
                    AnyBuffer::Mut(_) => Access::Rw,
                },
            });
        }

        Ok(Self {
            buffer: buffer,
            access,
            offset,
            length,
        })
    }
}

/// Runs a group of generated commands sequentially.
///
/// Input: closure containing generated command calls.
/// Returns the closure's result unchanged. This helper does not create threads
/// or parallelize commands; workgroup execution is controlled by the executor.
pub fn concurrent<T>(commands: impl FnOnce() -> T) -> T {
    commands()
}

/// Converts scalar fill values to the byte written by `fill`.
///
/// Input: scalar value.
/// Output: low byte used to fill a buffer range.
pub trait FillValue {
    /// Converts this scalar into a byte fill value.
    ///
    /// Input: `self`.
    /// Output: `u8` value passed to `write_bytes`.
    fn to_u8(self) -> u8;
}

macro_rules! impl_fill_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl FillValue for $ty {
                /// Converts this integer into the byte used by `fill`.
                ///
                /// Input: integer value.
                /// Output: low byte of the integer.
                fn to_u8(self) -> u8 {
                    self as u8
                }
            }
        )*
    };
}

impl_fill_value!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Fills a generated buffer range with one byte value.
///
/// # Safety
///
/// The pointer plus offset must describe `length` writable bytes within one
/// live allocation. The range must satisfy bounds even if constructed through
/// public fields, and writes must not conflict with other accesses. This
/// operation repeats the low byte of `value`, not its multi-byte representation.
///
/// # Errors
///
/// Returns [`Error::InvalidAccess`] for a read-only descriptor. This function
/// does not independently recheck numeric bounds or the mutable range's flag.
pub unsafe fn fill(target: AnyBufferRange, value: impl FillValue + Copy) -> Result<(), Error> {
    match target.buffer {
        AnyBuffer::Ref(_) => {
            return Err(Error::InvalidAccess {
                access: target.access,
                required: Access::Rw,
            });
        }
        AnyBuffer::Mut(buffer) => unsafe {
            core::ptr::write_bytes(buffer.ptr.add(target.offset), value.to_u8(), target.length);
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mutable_buffer(capacity: usize) -> AnyBuffer {
        BufferMut {
            ptr: core::ptr::dangling_mut(),
            len: capacity,
        }
        .into()
    }

    #[test]
    fn validates_buffer_ranges() {
        let range = AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 4, 8).unwrap();
        assert_eq!(range.offset, 4);
        assert_eq!(range.length, 8);

        assert!(AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 4, 0).is_ok());
        assert!(matches!(
            AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 17, 0),
            Err(Error::BufferRangeOutOfBounds { .. })
        ));
        assert!(matches!(
            AnyBufferRange::try_new(mutable_buffer(16), Access::Rw, 8, 9),
            Err(Error::BufferRangeOutOfBounds { .. })
        ));
    }
}
