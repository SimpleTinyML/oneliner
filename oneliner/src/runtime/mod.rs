mod buffer;
mod interface;
mod microflow;
mod prediction;

#[cfg(feature = "iree-runtime")]
mod iree;

pub use buffer::{
    bind_static_input, concurrent, fill, read_static_output, tensor_ref_from_raw, Access,
    FillValue, TensorRange, TensorRef, TensorSource,
};
pub use interface::{Error, ModelArtifacts, ModelSource, Predict, Result};
pub use microflow::MicroflowModel;
pub use prediction::Prediction;

#[cfg(feature = "iree-runtime")]
pub use iree::{
    dispatch, dispatch_fn_from_library, iree_hal_executable_dispatch_state_v0_t,
    iree_hal_executable_environment_v0_t, iree_hal_executable_import_thunk_v0_t,
    iree_hal_executable_import_v0_t, iree_hal_executable_library_header_t,
    iree_hal_executable_library_query_fn_t, iree_hal_executable_workgroup_state_v0_t,
    iree_hal_processor_v0_t, try_dispatch, DispatchFn,
};

#[cfg(feature = "iree-runtime")]
pub use aligned::{Aligned, A2, A4, A16, A32, A64};
pub type AlignedType = A64;