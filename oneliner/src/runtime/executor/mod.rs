#[cfg(feature = "ariel-os")]
pub use oneliner_ariel_os_executor::ArielOsExecutor;
pub use oneliner_executor::{Executor, SequentialExecutor, WorkItem};

#[cfg(feature = "ariel-os")]
pub type DefaultExecutor = ArielOsExecutor;

#[cfg(not(feature = "ariel-os"))]
pub type DefaultExecutor = SequentialExecutor;
