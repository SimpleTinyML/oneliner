#[cfg(feature = "ariel-os")]
pub use oneliner_ariel_os_executor::DefaultExecutor;
pub use oneliner_executor::{Executor, SequentialExecutor, WorkItem};

#[cfg(not(feature = "ariel-os"))]
pub type DefaultExecutor = SequentialExecutor;
