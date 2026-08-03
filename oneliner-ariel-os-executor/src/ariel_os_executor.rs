use ariel_os::log::debug;
use ariel_os::thread::sync::{Channel, Mutex};
use heapless::Deque;
use portable_atomic::{AtomicUsize, Ordering};

use oneliner_executor::{Executor, WorkItem};

const TASK_SLOT_COUNT: usize = 4;

static JOB_REMAINING: AtomicUsize = AtomicUsize::new(0);
static TASKS: Channel<usize> = Channel::new();
static TASK_QUEUE: Mutex<Deque<WorkItem, TASK_SLOT_COUNT>> = Mutex::new(Deque::new());

/// Ariel OS executor backed by a small fixed worker pool.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArielOsExecutor;

impl ArielOsExecutor {
    pub const fn new() -> Self {
        Self
    }
}

impl Executor for ArielOsExecutor {
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
    worker_loop();
}

#[ariel_os::thread(autostart)]
fn oneliner_ariel_os_worker_1() {
    worker_loop();
}
