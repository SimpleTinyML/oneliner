//! General interfaces for model inference and tensor manipulation.

use super::{Access, Aligned, AlignedType};

#[cfg(feature = "ndarray")]
use ndarray::{ArrayView4, ArrayViewMut4};

/// Shape for 4D Tensor.
pub type Shape4D = (usize, usize, usize, usize);

/// Shape alias used by the public tensor API.
pub type Shape = Shape4D;

/// Nested array storage for tensor.
pub type TensorArray<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> =
    [[[[T; D4]; D3]; D2]; D1];

/// Four-dimensional tensor backed by an aligned, fixed-size nested array.
pub struct Tensor4D<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> {
    storage: Aligned<AlignedType, TensorArray<T, D1, D2, D3, D4>>,
}

/// Public alias for the aligned, owned [`Tensor4D`] storage type.
pub type Tensor<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize> =
    Tensor4D<T, D1, D2, D3, D4>;

impl<T, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    Tensor4D<T, D1, D2, D3, D4>
{
    /// The four compile-time determined sizes of dimentions.
    pub const SHAPE: Shape = (D1, D2, D3, D4);

    /// Number of elements.
    pub const LEN: usize = D1 * D2 * D3 * D4;

    /// Creates inline storage with every element initialized to `value`.
    pub const fn new(value: T) -> Self
    where
        T: Copy,
    {
        Self {
            storage: Aligned([[[[value; D4]; D3]; D2]; D1]),
        }
    }

    /// Moves a nested array into aligned tensor storage.
    pub const fn from_array(storage: TensorArray<T, D1, D2, D3, D4>) -> Self {
        Self {
            storage: Aligned(storage),
        }
    }

    /// Borrows all elements in contiguous storage order.
    pub fn as_slice(&self) -> &[T] {
        self.storage[..]
            .as_flattened()
            .as_flattened()
            .as_flattened()
    }

    /// Mutably borrows all elements in contiguous storage order.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.storage[..]
            .as_flattened_mut()
            .as_flattened_mut()
            .as_flattened_mut()
    }

    /// Returns a element pointer without extending its lifetime.
    ///
    /// Moving or dropping the tensor can invalidate the pointer. Dereferencing it
    /// requires the usual Rust validity and aliasing guarantees.
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Returns a mutable element pointer without extending its lifetime.
    ///
    /// The caller must preserve exclusive access while using it for writes.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_slice_mut().as_mut_ptr()
    }

    /// Returns the element payload size in bytes, excluding alignment padding.
    pub fn byte_len(&self) -> usize {
        core::mem::size_of_val(self.as_slice())
    }

    /// Returns the number of elements, not the number of bytes.
    pub const fn len(&self) -> usize {
        Self::LEN
    }

    /// Returns 0 if at least one axis is zero.
    pub const fn is_empty() -> bool {
        Self::LEN == 0
    }

    
    /// Returns the dimension of tensor.
    pub const fn dim() -> usize {
        4
    }

    /// Returns the shape of tensor.
    pub const fn shape() -> Shape {
        Self::SHAPE
    }



    #[cfg(feature = "ndarray")]
    /// Borrows the storage as a contiguous ndarray view (`ndarray` feature).
    ///
    /// Constructing this view does not allocate. The ndarray dependency itself
    /// uses the allocation library.
    ///
    /// # Examples
    ///
    /// ```
    /// use oneliner_runtime::Tensor;
    /// let tensor = Tensor::<u8, 1, 1, 2, 2>::new(7);
    /// assert_eq!(tensor.view()[[0, 0, 1, 1]], 7);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if ndarray rejects the shape, including its platform size limits.
    pub fn view(&self) -> ArrayView4<'_, T> {
        ArrayView4::from_shape(Self::SHAPE, self.as_slice())
            .expect("Tensor4D shape must match its storage")
    }

    #[cfg(feature = "ndarray")]
    /// Mutably borrows storage as an ndarray view (`ndarray` feature).
    ///
    /// # Panics
    ///
    /// Panics if ndarray rejects the shape, including its platform size limits.
    pub fn view_mut(&mut self) -> ArrayViewMut4<'_, T> {
        ArrayViewMut4::from_shape(Self::SHAPE, self.as_slice_mut())
            .expect("Tensor4D shape must match its storage")
    }

    #[cfg(feature = "ndarray")]
    /// Replaces every element with a clone of `value`.
    ///
    /// # Panics
    ///
    /// Propagates a panic from `T::clone`. With `ndarray` enabled, also propagates
    /// a shape-validation panic from `view_mut`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.view_mut().fill(value);
    }

    #[cfg(not(feature = "ndarray"))]
    /// Replaces every element with a clone of `value`.
    ///
    /// # Panics
    ///
    /// Propagates a panic from `T::clone`. With `ndarray` enabled, also propagates
    /// a shape-validation panic from `view_mut`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.as_slice_mut().fill(value);
    }
}

/// Model inference trait implemented directly by generated model instances.
///
/// Import this trait to call `model.run(...)` or `Model::create_input_tensor()`.
/// Generated models have one input and one output tensor; their tensor types are
/// [`Tensor`]. `create_input_tensor` creates an input tensor that is suitable for model.
/// User pre-fills the input tensor via [`Tensor`]'s method before model inference.
/// `run` executes model inference with the input tensor borrowed from the caller and returns output tensor filled with inference results.
///
/// # Examples
///
/// This small Rust implementation demonstrates the trait without a compiler
/// toolchain.
///
/// ```
/// use oneliner_runtime::{ModelInference, Tensor};
/// struct Echo;
/// impl ModelInference for Echo {
///     type InputTensor = Tensor<u8, 1, 1, 1, 2>;
///     type OutputTensor = Self::InputTensor;
///     fn create_input_tensor() -> Self::InputTensor { Tensor::new(0) }
///     fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor {
///         let mut output = Self::create_input_tensor();
///         output.as_slice_mut().copy_from_slice(input.as_slice());
///         output
///     }
/// }
/// let mut model = Echo;
/// let mut input = Echo::create_input_tensor();
/// input.as_slice_mut().copy_from_slice(&[3, 7]);
/// assert_eq!(model.run(&input).as_slice(), &[3, 7]);
/// ```
pub trait ModelInference {
    /// Input tensor accepted by this model. Type instance of [`Tensor`].
    type InputTensor;
    /// Output tensor produced by this model. Type instance of [`Tensor`].
    type OutputTensor;

    /// Runs inference and returns the results encoded in output tensor.
    ///
    /// The trait has no recoverable-error return value; custom implementations
    /// must document their own failure behavior.
    fn run(&mut self, input: &Self::InputTensor) -> Self::OutputTensor;

    /// Creates a zero-filled input tensor with the model's element type and dimensions.
    fn create_input_tensor() -> Self::InputTensor;
}

/// Metadata exposed by a model generated with `#[model]`. Useful for debug and trace-back.
pub trait ModelSource {
    /// Path of source model file on the build host.
    const MODEL_PATH: &'static str;
    /// Model metadata including paths of intermediate files, estimated memory usage etc.
    const ARTIFACTS: ModelArtifacts;
}

/// Failures reported by raw buffer validation and backend dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The binding list exceeds runtime descriptor capacity.
    TooManyBindings {
        /// Number of descriptors or constants supplied.
        provided: usize,
        /// Maximum permitted count or byte capacity.
        capacity: usize,
    },
    /// The push-constant count exceeds its ABI representation.
    TooManyConstants {
        /// Number of descriptors or constants supplied.
        provided: usize,
        /// Maximum permitted count or byte capacity.
        capacity: usize,
    },
    /// The queried library or requested export is unavailable.
    MissingDispatchFunction {
        /// Zero-based requested export index.
        ordinal: usize,
    },
    /// A workgroup dimension exceeds its ABI representation.
    WorkgroupCountTooLarge {
        /// Workgroup axis, such as `z`.
        dimension: char,
        /// Requested workgroup count.
        value: u32,
    },
    /// The byte range overflows or exceeds the declared buffer length.
    BufferRangeOutOfBounds {
        /// Start of the range in bytes.
        offset: usize,
        /// Length of the range in bytes.
        length: usize,
        /// Maximum permitted count or byte capacity.
        capacity: usize,
    },
    /// A backend workgroup returned a nonzero status.
    DispatchFailed {
        /// Backend status code.
        status: i32,
    },
    /// The descriptor kind is incompatible with the requested access.
    InvalidAccess {
        /// Access flag carried by the range.
        access: Access,
        /// Access expected for this operation.
        required: Access,
    },
    /// A buffer constructor received a null pointer.
    NullPointer,
}

impl core::fmt::Display for Error {
    /// Formats a runtime error for diagnostics.
    ///
    /// Input: formatter supplied by Rust formatting machinery.
    /// Output: `Ok(())` when the message was written.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooManyBindings { provided, capacity } => write!(
                f,
                "dispatch has {provided} bindings, but runtime stack storage holds {capacity}"
            ),
            Self::TooManyConstants { provided, capacity } => write!(
                f,
                "dispatch has {provided} constants, but the IREE ABI supports {capacity}"
            ),
            Self::MissingDispatchFunction { ordinal } => {
                write!(f, "IREE library does not expose dispatch ordinal {ordinal}")
            }
            Self::WorkgroupCountTooLarge { dimension, value } => write!(
                f,
                "workgroup count {dimension}={value} exceeds the IREE ABI limit"
            ),
            Self::BufferRangeOutOfBounds {
                offset,
                length,
                capacity,
            } => write!(
                f,
                "tensor range offset={offset} length={length} exceeds capacity {capacity}"
            ),
            Self::DispatchFailed { status } => {
                write!(f, "backend dispatch returned status {status}")
            }

            Self::InvalidAccess { access, required } => write!(
                f,
                "tensor access {access:?} is invalid; expected {required:?}"
            ),
            Self::NullPointer => write!(f, "tensor pointer is null"),
        }
    }
}

/// Artifact paths on the host of compiler and memory usage estimation of model inference.
///
/// Memory usage sizes describe generated
/// resources and object sections before final application linking. They omit
/// application/OS state, stack usage and allocator overhead. Alignment and final
/// linker placement can change the actual footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelArtifacts {
    /// Backend name, currently `iree`.
    pub backend: &'static str,
    /// Original model file path.
    pub model_path: &'static str,
    /// MLIR input passed to IREE after frontend conversion, if any.
    pub compile_input_path: &'static str,
    /// Native model ELF object emitted by IREE.
    pub object_path: &'static str,
    /// Native link input; currently the same path as `object_path`.
    pub link_path: &'static str,
    /// IREE phase dump MLIRs by the Stream/Flow converter.
    pub ir_path: &'static str,
    /// Generated Rust execution source by the Stream/Flow converter.
    pub flow_rs_path: &'static str,
    /// JSON resource/command metadata used by [`model`] macro expansion.
    pub metadata_json_path: &'static str,
    /// Size in bytes of the model's input tensor.
    pub input_size: usize,
    /// Size in bytes of the model's output tensor.
    pub output_size: usize,
    /// Model parameter/weight bytes.
    pub params_size: usize,
    /// Machine code bytes of the compiled model.
    pub code_size: usize,
    /// Read-only data bytes embedded in the compiled model object (lookup
    /// tables, library metadata, unwind tables).
    pub rodata_size: usize,
    /// Sum of `params_size`, `code_size` and `rodata_size` of the compiled model before final linking.
    pub total_flash_size: usize,
    /// Workspace RAM usage of the model.
    pub ram_size: usize,
}
