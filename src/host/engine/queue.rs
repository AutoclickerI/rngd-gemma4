//! Queue device launches in FIFO order to keep the device busy while the host prepares later launches.
//!
//! The launch thread polls `Function::run` in submission order. Its first poll submits and can spin
//! for up to 200 µs. The host reaps completed launches before posting more work. The queue uses
//! the runtime's 16 answer slots per cluster link when enabled.
//!
//! A host transfer proceeds alongside launches. So a host read of a buffer that a queued launch writes, and a
//! host write of a buffer that a queued launch reads, must follow `drain`. The engine queues launches from the
//! step's embedding to `last`, after the step's table uploads, and drains before media encoding, sampler readback and at the step boundary. The ≤ 2 KB posts `last` needs (its
//! output rows) go out while segments are queued; no queued launch reads them.
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use furiosa_opt_std::prelude::*;

/// A launch completion or a FIFO fence.
pub type Run = Pin<Box<dyn Future<Output = Result<(), Error>>>>;

thread_local! {
    static QUEUE: RefCell<VecDeque<Run>> = const { RefCell::new(VecDeque::new()) };
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static FUNCTIONS: RefCell<std::collections::HashMap<(&'static str, Vec<usize>), std::sync::Arc<furiosa_opt_std::backend::npu::Function>>> =
        RefCell::new(std::collections::HashMap::new());
    /// Lookup by registry path address and short axis keys without an allocation.
    static FAST: RefCell<std::collections::HashMap<(usize, usize, [usize; 4]), std::sync::Arc<furiosa_opt_std::backend::npu::Function>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Enable queued launches on this thread. When disabled, launches are awaited.
pub fn set_enabled(enabled: bool) {
    ENABLED.set(enabled);
}

/// Whether launches queue on this thread.
pub fn queueing() -> bool {
    ENABLED.get()
}

fn len() -> usize {
    QUEUE.with_borrow(|q| q.len())
}

/// Entries remaining after the most recent reap.
pub fn queued_len() -> usize {
    len()
}

fn pop_front() -> Option<Run> {
    QUEUE.with_borrow_mut(|q| q.pop_front())
}

/// Polls `run` once: its result if it finished in that poll.
async fn poll_once(run: &mut Run) -> Option<Result<(), Error>> {
    std::future::poll_fn(|cx| {
        Poll::Ready(match run.as_mut().poll(cx) {
            Poll::Ready(result) => Some(result),
            Poll::Pending => None,
        })
    })
    .await
}

/// Takes the answered launches off the front (the device answers in submission order).
pub(super) async fn reap() -> Result<(), Error> {
    while let Some(mut front) = pop_front() {
        match poll_once(&mut front).await {
            Some(result) => result?,
            None => {
                QUEUE.with_borrow_mut(|q| q.push_front(front));
                break;
            }
        }
    }
    Ok(())
}

/// Waits for every queued launch, oldest first; the first failure ends the wait and drops the rest
/// (a dropped launch collects its answer, rt `Launch::drop`).
pub async fn drain() -> Result<(), Error> {
    flush_pending().await?;
    drain_submitted().await
}

async fn drain_submitted() -> Result<(), Error> {
    while let Some(front) = pop_front() {
        if let Err(why) = front.await {
            clear();
            return Err(why);
        }
    }
    Ok(())
}

/// Drops every queued launch (after a failure).
pub fn clear() {
    crate::host::engine::step_image::launch_boundary::clear();
    crate::host::engine::step_image::persist::clear();
    let dropped: Vec<Run> = QUEUE.with_borrow_mut(|q| q.drain(..).collect());
    drop(dropped);
}

/// Choose when to prepare the next step between launches from the worker's run delay.
#[derive(Clone, Copy)]
struct GateRule {
    steps_on: u32,
    steps_off: u32,
    off_above: f64,
    on_below: f64,
}

/// Disable early preparation at 2% run delay and re-enable it below 1%.
const GATE_RULE: GateRule = GateRule { steps_on: 32, steps_off: 16, off_above: 0.02, on_below: 0.01 };
/// The fewest unthrottled steps a decision rests on.
const GATE_MIN_UNTHROTTLED: u32 = 8;

thread_local! {
    /// Suppress the machine gate during sections that control next-step preparation.
    static NAP_SUSPENDED: Cell<bool> = const { Cell::new(false) };
}

/// Suspend or resume the machine gate on this thread.
pub fn suspend_nap(on: bool) {
    NAP_SUSPENDED.set(on);
}

/// Whether the machine gate permits early preparation between launches.
pub fn napping() -> bool {
    nap_on()
}

fn nap_on() -> bool {
    !NAP_SUSPENDED.get() && NAP_GATE.with_borrow(|g| g.on)
}

/// The machine gate's decision window.
#[derive(Default)]
struct NapGate {
    on: bool,
    /// At the last step's end: (time, worker run delay ns, cgroup nr_throttled).
    last: Option<(std::time::Instant, u64, u64)>,
    steps: u32,
    unthrottled: u32,
    wall_ns: u64,
    delay_ns: u64,
}

thread_local! {
    static NAP_GATE: RefCell<NapGate> = RefCell::new(NapGate::default());
}

/// Whether a window of `steps` steps with `unthrottled` unthrottled ones decides now under `rule` with the nap `on`,
/// and if so what it decides from the run-delay `share` (the window's delay over its unthrottled wall).
fn gate_decide(rule: &GateRule, on: bool, steps: u32, unthrottled: u32, share: f64) -> Option<bool> {
    // A decision rests on at least `GATE_MIN_UNTHROTTLED` unthrottled steps (the window grows until it has them);
    // a window of `4 * steps_on` steps that still lacks them means the quota binds nearly always: nap.
    let enough = unthrottled >= GATE_MIN_UNTHROTTLED;
    let window = if on { rule.steps_on } else { rule.steps_off };
    if (steps >= window && enough) || steps >= 4 * rule.steps_on {
        Some(if !enough { true } else { share < if on { rule.off_above } else { rule.on_below } })
    } else {
        None
    }
}

/// Books the step just ended into one gate machine and decides when its window is full.
fn gate_book(g: &mut NapGate, rule: &GateRule, step: u64, now: std::time::Instant, throttles: u64, delay: u64) {
    if let Some((at, d, n)) = g.last {
        g.steps += 1;
        if throttles == n {
            g.unthrottled += 1;
            g.wall_ns += now.duration_since(at).as_nanos() as u64;
            g.delay_ns += delay.saturating_sub(d);
        }
    }
    g.last = Some((now, delay, throttles));
    let share = g.delay_ns as f64 / g.wall_ns.max(1) as f64;
    if let Some(on) = gate_decide(rule, g.on, g.steps, g.unthrottled, share) {
        eprintln!(
            "ENGINE_NAP_GATE step {step} {}{} worker run delay {:.3}% over {} unthrottled of {} steps",
            if on { "on" } else { "off" },
            if on != g.on { " (switched)" } else { "" },
            100.0 * share,
            g.unthrottled,
            g.steps
        );
        *g = NapGate { on, last: g.last, ..NapGate::default() };
    }
}

/// Updates the machine gate after each step.
pub fn nap_gate_step(step: u64) {
    let now = std::time::Instant::now();
    let (_, throttles, _) = cgroup_now();
    let (_, delay) = self_sched();
    NAP_GATE.with_borrow_mut(|g| gate_book(g, &GATE_RULE, step, now, throttles, delay));
}

/// Forgets the loaded functions (a new engine may run on a new device) and the table aliases (`uploads::alias`).
pub fn reset() {
    clear();
    crate::host::engine::uploads::clear_aliases();
    FUNCTIONS.with_borrow_mut(|f| f.clear());
    FAST.with_borrow_mut(|f| f.clear());
}

/// The device function at registry `path` and `key` (a generic fn's axis sizes; empty for a concrete fn),
/// loaded once per engine.
async fn function(device: &Device, path: &'static str, key: &[usize]) -> Result<std::sync::Arc<furiosa_opt_std::backend::npu::Function>, Error> {
    let fast = (key.len() <= 4).then(|| {
        let mut k = [usize::MAX; 4];
        k[..key.len()].copy_from_slice(key);
        (path.as_ptr() as usize, path.len(), k)
    });
    if let Some(fast) = fast
        && let Some(f) = FAST.with_borrow(|m| m.get(&fast).cloned())
    {
        return Ok(f);
    }
    let f = match FUNCTIONS.with_borrow(|f| f.get(&(path, key.to_vec())).cloned()) {
        Some(f) => f,
        None => {
            let f = std::sync::Arc::new(furiosa_opt_std::backend::npu::function(device, path, key).await?);
            FUNCTIONS.with_borrow_mut(|m| m.insert((path, key.to_vec()), f.clone()));
            f
        }
    };
    if let Some(fast) = fast {
        FAST.with_borrow_mut(|m| m.insert(fast, f.clone()));
    }
    Ok(f)
}

/// A raw launch (the DSL and fused programs bind their own buffer regions) through the queue. A table tensor whose
/// bytes went out inside the words upload binds that region (`uploads::substitute`, `pack`). A step family's member
/// runs its family image (`step_image::route`).
pub async fn run(
    function: std::sync::Arc<furiosa_opt_std::backend::npu::Function>,
    mut inputs: furiosa_opt_std::runtime::Buffers,
    mut outputs: furiosa_opt_std::runtime::Buffers,
) -> Result<(), Error> {
    crate::host::engine::uploads::substitute(&mut inputs);
    crate::host::engine::uploads::substitute(&mut outputs);
    let mut ready = crate::host::engine::step_image::launch_boundary::flush();
    if crate::host::engine::step_image::launch_boundary::accepts(&function, &inputs, &outputs) {
        ready.extend(crate::host::engine::step_image::persist::flush());
        submit_ready(ready).await?;
        crate::host::engine::step_image::launch_boundary::hold((function, inputs, outputs));
        return Ok(());
    }
    ready.extend(crate::host::engine::step_image::persist::enqueue(function, inputs, outputs));
    submit_ready(ready).await
}

/// Submit a composed Full-attention launch after any pending RSF and RSS launches.
pub(crate) async fn run_full_sequence(
    function: std::sync::Arc<furiosa_opt_std::backend::npu::Function>,
    mut inputs: furiosa_opt_std::runtime::Buffers,
    mut outputs: furiosa_opt_std::runtime::Buffers,
) -> Result<(), Error> {
    crate::host::engine::uploads::substitute(&mut inputs);
    crate::host::engine::uploads::substitute(&mut outputs);
    let identity = std::sync::Arc::as_ptr(&function) as usize;
    let mut ready = crate::host::engine::step_image::launch_boundary::flush();
    ready.extend(crate::host::engine::step_image::persist::enqueue(function, inputs, outputs));
    let last = ready.last().expect("a family function is a direct launch, never an RSS member");
    assert_eq!(std::sync::Arc::as_ptr(&last.function) as usize, identity);
    submit_ready(ready).await
}

async fn submit_ready(ready: Vec<crate::host::engine::step_image::persist::Launch>) -> Result<(), Error> {
    for launch in ready {
        reap().await?;
        while len() >= if queueing() { crate::host::engine::launch_thread::MAX_IN_FLIGHT } else { 1 } {
            pop_front().expect("queued helper completion").await?;
        }
        let completion = crate::host::engine::launch_thread::submit(
            launch.function, launch.inputs, launch.outputs, launch.owners,
        );
        if queueing() {
            QUEUE.with_borrow_mut(|q| q.push_back(Box::pin(completion)));
        } else {
            completion.await?;
        }
    }
    Ok(())
}

/// Flush collected RSS frames before a host read, a step boundary or a table reuse.
pub async fn flush_pending() -> Result<(), Error> {
    submit_ready(crate::host::engine::step_image::persist::flush()).await?;
    submit_ready(crate::host::engine::step_image::launch_boundary::flush()).await?;
    Ok(())
}

/// A typed launch of the `#[device]` fn at `path` and `key` on `args` (its arguments after the leading
/// `&mut Device`, which binds nothing) through the queue. Bind the buffers before entering the FIFO.
pub async fn launch<Args: furiosa_opt_std::runtime::DeviceSend>(device: &Device, path: &'static str, key: &[usize], args: Args) -> Result<(), Error> {
    let function = function(device, path, key).await?;
    let mut inputs = furiosa_opt_std::runtime::Buffers::new();
    args.bind(&mut inputs)?;
    drop(args);
    let result = run(function, inputs, furiosa_opt_std::runtime::Buffers::new()).await;
    result
}

/// `queued!(device, ops_core::f, (args..))`: `launch(ops_core::f, (&mut *device, args..)).await` for a concrete
/// device fn returning `()`. A generic
/// fn names its instantiation's key (its axis sizes in declaration order): `queued!(device, f, [R::SIZE], (..))`.
/// The compiler discovers the kernel through the typed launch when the queue is disabled.
macro_rules! queued {
    ($device:expr, $function:expr, ($($arg:expr),* $(,)?) $(,)?) => {
        $crate::host::engine::queue::queued!($device, $function, [], ($($arg),*))
    };
    ($device:expr, $function:expr, [$($key:expr),* $(,)?], ($($arg:expr),* $(,)?) $(,)?) => {{
        let launched = if $crate::host::engine::queue::queueing() {
            $crate::host::engine::queue::launch(&*$device, $function.path(), &[$(($key) as usize),*], ($($arg,)*)).await
        } else {
            ::furiosa_opt_std::runtime::launch($function, (&mut *$device, $($arg,)*)).await
        };
        launched
    }};
}
pub(crate) use queued;

/// A FIFO completion point, independent of launches queued after it.
#[derive(Clone)]
pub(crate) struct Fence(std::sync::Arc<Cell<Option<std::time::Instant>>>);

pub(crate) fn fence() -> Fence {
    let done = std::sync::Arc::new(Cell::new((len() == 0).then(std::time::Instant::now)));
    if done.get().is_none() {
        let mark = done.clone();
        QUEUE.with_borrow_mut(|q| q.push_back(Box::pin(async move { mark.set(Some(std::time::Instant::now())); Ok(()) })));
    }
    Fence(done)
}

pub(crate) async fn drain_through(fence: &Fence) -> Result<(), Error> {
    while fence.0.get().is_none() {
        let front = pop_front().expect("a live fence remains in the FIFO");
        if let Err(why) = front.await { clear(); return Err(why); }
    }
    Ok(())
}

use std::os::unix::fs::FileExt;

thread_local! {
    static CGROUP: Option<std::fs::File> = std::fs::File::open("/sys/fs/cgroup/cpu.stat").ok();
    static SELF_SCHED: Option<std::fs::File> = std::fs::File::open("/proc/thread-self/schedstat").ok();
}

fn read_at(file: &std::fs::File, buf: &mut [u8]) -> usize {
    file.read_at(buf, 0).unwrap_or(0)
}

/// The cgroup's (usage µs, nr_throttled, throttled µs).
pub(crate) fn cgroup_now() -> (u64, u64, u64) {
    CGROUP.with(|f| {
        let Some(f) = f else { return (0, 0, 0) };
        let mut buf = [0u8; 512];
        let n = read_at(f, &mut buf);
        let s = std::str::from_utf8(&buf[..n]).unwrap_or("");
        let field = |name: &str| s.lines().find_map(|l| l.strip_prefix(name)?.trim().parse::<u64>().ok()).unwrap_or(0);
        (field("usage_usec"), field("nr_throttled"), field("throttled_usec"))
    })
}

/// The calling thread's (CPU ns, run delay ns).
pub(crate) fn self_sched() -> (u64, u64) {
    SELF_SCHED.with(|f| {
        let Some(f) = f else { return (0, 0) };
        let mut buf = [0u8; 128];
        let n = read_at(f, &mut buf);
        let mut it = std::str::from_utf8(&buf[..n]).unwrap_or("").split_whitespace().map(|v| v.parse::<u64>().unwrap_or(0));
        (it.next().unwrap_or(0), it.next().unwrap_or(0))
    })
}
