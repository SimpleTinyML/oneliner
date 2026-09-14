//! Model workload scheduler and executor for Ariel OS, supporting concurrent execution on multicore MCUs.

use ariel_os::log::debug;
use ariel_os::thread::sync::{Channel, Mutex};
use ariel_os::thread::CORE_COUNT;
use heapless::Deque;
use portable_atomic::{AtomicUsize, Ordering};

use oneliner_executor::{Executor, WorkItem};

const TASK_SLOT_COUNT: usize = 4;

// These resources are shared by all executor handles and model instances.
static JOB_REMAINING: AtomicUsize = AtomicUsize::new(0);
static TASKS: Channel<usize> = Channel::new();
static TASK_QUEUE: Mutex<Deque<WorkItem, TASK_SLOT_COUNT>> = Mutex::new(Deque::new());

/// Default executor for Ariel OS.
///
/// On multi-core MCUs (`CORE_COUNT > 1`) this is [`ArielOsExecutor`] with
/// `MULTICORE = true` and schedules work on a small groups of worker attaching 
/// to different cores. On single-core MCUs it falls back to `SequentialExecutor` implementation,
/// which runs work items sequentially in submission order.

pub type DefaultExecutor = ArielOsExecutor<{ CORE_COUNT > 1 }>;

/// Ariel OS executor.
///
/// When `MULTICORE` is `true` work items are executed by a small fixed worker
/// pool scheduled by Ariel OS. When `MULTICORE` is `false` work items run
/// sequentially in submission order, mirroring the `SequentialExecutor`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArielOsExecutor<const MULTICORE: bool>;

impl<const MULTICORE: bool> ArielOsExecutor<MULTICORE> {
    /// Creates a singleton handle to the execution facilities.
    ///
    /// Worker threads are registered by the Ariel OS's autostart macro;
    /// this constructor does not allocate memory or create threads.
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for ArielOsExecutor<true> {
    fn schedule(&mut self, item: WorkItem) {
        loop {
            let result = {
                let mut queue = TASK_QUEUE.lock();
                queue.push_back(item)
            };
            match result {
                Ok(()) => {
                    JOB_REMAINING.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                Err(_) => ariel_os::thread::yield_same(),
            }
        }

        TASKS.send(&42);
    }

    fn wait_job_completion(&mut self) {
        while JOB_REMAINING.load(Ordering::Acquire) > 0 {
            ariel_os::thread::yield_same();
        }
    }
}

impl Executor for ArielOsExecutor<false> {
    fn schedule(&mut self, item: WorkItem) {
        item.run();
    }

    fn wait_job_completion(&mut self) {}
}

fn worker_loop() -> ! {
    let thread_id = ariel_os::thread::current_tid().unwrap();
    let core = ariel_os::thread::core_id();
    debug!("[{:?}] Running on core {:?}", thread_id, core);

    loop {
        let _signal = TASKS.recv();

        let task = {
            let mut queue = TASK_QUEUE.lock();
            queue.pop_front()
        };
        if let Some(task) = task {
            task.run();
            JOB_REMAINING.fetch_sub(1, Ordering::Release);
        }
    }
}

#[ariel_os::thread(autostart)]
fn oneliner_ariel_os_worker_0() {
    if CORE_COUNT <= 1 {
        return;
    }
    worker_loop();
}

#[ariel_os::thread(autostart)]
fn oneliner_ariel_os_worker_1() {
    if CORE_COUNT <= 1 {
        return;
    }
    worker_loop();
}
