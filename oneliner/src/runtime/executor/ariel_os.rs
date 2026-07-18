use core::mem::MaybeUninit;

use ariel_os::thread::sync::Channel;

use super::{Executor, WorkItem};

use log::{debug, trace};

pub const ARIEL_OS_EXECUTOR_WORKER_STACK_SIZE: usize = 2048;
pub const ARIEL_OS_EXECUTOR_WORKER_PRIORITY: u8 = 1;

#[derive(Clone, Copy)]
struct TaskMessage {
    run: unsafe fn(usize),
    slot: usize,
    completion: usize,
}

static TASKS: Channel<TaskMessage> = Channel::new();

/// Ariel OS executor backed by a small fixed worker pool.
///
/// Work items are sent to Ariel OS worker threads and `schedule` waits until
/// the selected worker has written the result back to the caller's stack frame.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArielOsExecutor;

impl ArielOsExecutor {
    /// Creates an Ariel OS executor handle.
    ///
    /// Input: none.
    /// Output: stateless executor handle using the global worker pool.
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for ArielOsExecutor {
    /// Runs the work item on an Ariel OS worker thread.
    ///
    /// Input: work item to execute.
    /// Output: result returned by the work item.
    fn schedule<W>(&mut self, item: W) -> W::Output
    where
        W: WorkItem,
    {
        let completion: Channel<()> = Channel::new();
        let mut slot = TaskSlot {
            item: Some(item),
            output: MaybeUninit::uninit(),
        };
        let task = TaskMessage {
            run: run_task::<W>,
            slot: core::ptr::addr_of_mut!(slot) as usize,
            completion: core::ptr::addr_of!(completion) as usize,
        };

        TASKS.send(&task);
        completion.recv();
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Acquire);

        unsafe { slot.output.assume_init() }
    }
}

struct TaskSlot<W>
where
    W: WorkItem,
{
    item: Option<W>,
    output: MaybeUninit<W::Output>,
}

unsafe fn run_task<W>(slot: usize)
where
    W: WorkItem,
{
    let slot = unsafe { &mut *(slot as *mut TaskSlot<W>) };
    let item = slot
        .item
        .take()
        .expect("Ariel OS executor task missing item");
    slot.output.write(item.run());
}

fn worker_loop() -> ! {
    loop {
        let task = TASKS.recv();
        trace!("Worker received task: run={:p}, slot={:#x}, completion={:#x}", task.run, task.slot, task.completion);
        unsafe {
            (task.run)(task.slot);
            let completion = &*(task.completion as *const Channel<()>);
            completion.send(&());
        }
    }
}

fn oneliner_ariel_os_worker_0() {
    debug!("Starting Ariel OS worker 0");
    worker_loop();
}

fn oneliner_ariel_os_worker_1() {
    debug!("Starting Ariel OS worker 1");
    worker_loop();
}

ariel_os::thread::autostart_thread!(
    oneliner_ariel_os_worker_0,
    stacksize = ARIEL_OS_EXECUTOR_WORKER_STACK_SIZE,
    priority = ARIEL_OS_EXECUTOR_WORKER_PRIORITY,
    affinity = None
);

ariel_os::thread::autostart_thread!(
    oneliner_ariel_os_worker_1,
    stacksize = ARIEL_OS_EXECUTOR_WORKER_STACK_SIZE,
    priority = ARIEL_OS_EXECUTOR_WORKER_PRIORITY,
    affinity = None
);
