//! Bounded, ordered helper for raw device launches. Only function handles and owned
//! buffers cross threads. Completion (including drop) retires them before a host
//! transfer can pass `queue::drain`.
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use furiosa_opt_std::Error;
use furiosa_opt_std::{backend::npu::Function, runtime::Buffers};

/// Number of runtime answer slots per cluster link.
pub(super) const MAX_IN_FLIGHT: usize = 16;
#[derive(Default)]
struct State { done: bool, result: Option<Result<(), Error>>, waker: Option<Waker> }
#[derive(Default)]
struct Answer { state: Mutex<State>, ready: Condvar }
impl Answer {
    fn complete(&self, result: Result<(), Error>) {
        let mut state = self.state.lock().unwrap();
        state.result = Some(result);
        state.done = true;
        if let Some(waker) = state.waker.take() { waker.wake(); }
        self.ready.notify_all();
    }
}
pub struct Completion(Arc<Answer>);
impl Future for Completion {
    type Output = Result<(), Error>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.0.state.lock().unwrap();
        if let Some(result) = state.result.take() { return Poll::Ready(result); }
        state.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        // Error cleanup cannot detach still-running launches from their buffers.
        let mut state = self.0.state.lock().unwrap();
        while !state.done { state = self.0.ready.wait(state).unwrap(); }
    }
}
struct Command {
    function: Arc<Function>, inputs: Buffers, outputs: Buffers, owners: Vec<Buffers>,
    answer: Arc<Answer>, claimed: bool,
}
struct Pending {
    run: Pin<Box<dyn Future<Output = Result<(), Error>>>>,
    answer: Arc<Answer>, failed: bool,
}
impl Pending {
    fn poll(&mut self, cx: &mut Context<'_>) -> bool {
        let result = self.run.as_mut().poll(cx);
        if let Poll::Ready(result) = result {
            self.failed = result.is_err();
            // Replace/drop the future and its buffer owners before publishing completion.
            self.run = Box::pin(std::future::ready(Ok(())));
            self.answer.complete(result);
            true
        } else { false }
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        // Never free a submitted launch's buffers merely because its owner unwinds.
        // Device errors return through the future; a wedged device is stopped by the job watchdog.
        if !self.answer.state.lock().unwrap().done {
            let mut cx=Context::from_waker(Waker::noop());
            while self.run.as_mut().poll(&mut cx).is_pending(){std::thread::yield_now();}
            self.run=Box::pin(std::future::ready(Ok(())));
            self.answer.complete(Err(Error::Poisoned));
        }
    }
}
fn worker(rx: mpsc::Receiver<Command>) {
    // Keep polling sleeps precise; failure falls back to scheduler yields.
    unsafe extern "C" { fn prctl(option: i32, ...) -> i32; }
    let low_slack=unsafe { prctl(29,1000u64) }==0;
    if !low_slack { eprintln!("ENGINE_LAUNCH_THREAD timerslack: {}",std::io::Error::last_os_error()); }
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let _entered = rt.enter();
    let mut pending: VecDeque<Pending> = VecDeque::new();
    let mut cx = Context::from_waker(Waker::noop());
    let mut failed = false;
    let mut disconnected=false;
    loop {
        let mut submitted=false;
        // Fill available hardware slots before polling outstanding completions.
        while pending.len()<MAX_IN_FLIGHT && !disconnected {
            let next=if pending.is_empty() && !submitted {rx.recv().ok()} else {
                match rx.try_recv(){Ok(cmd)=>Some(cmd),Err(mpsc::TryRecvError::Empty)=>break,Err(mpsc::TryRecvError::Disconnected)=>None}
            };
            let Some(mut cmd)=next else {disconnected=true;break;};
            submitted=true;
            if failed {cmd.answer.complete(Err(Error::Poisoned));cmd.claimed=true;continue;}
            let (function,inputs,outputs,answer)=(cmd.function.clone(),std::mem::take(&mut cmd.inputs),std::mem::take(&mut cmd.outputs),cmd.answer.clone());
            let owners = std::mem::take(&mut cmd.owners);
            let run=Box::pin(async move {
                let result = function.run(&inputs,&outputs).await;
                drop(owners);
                result
            });
            let mut p=Pending{run,answer,failed:false};cmd.claimed=true;
            // This first poll both submits and starts waiting in rt0.9. Keep its
            // blocking spin on this thread, in FIFO order, while the host binds ahead.
            if p.poll(&mut cx){failed|=p.failed;}else{pending.push_back(p);}
        }
        while let Some(mut p)=pending.pop_front(){
            if p.poll(&mut cx){failed|=p.failed;}else{pending.push_front(p);break;}
        }
        if disconnected && pending.is_empty(){break;}
        if !submitted && !pending.is_empty(){
            // Four outstanding launches cover this poll interval. Keep the short
            // tail responsive and avoid monopolizing the launcher's shared mutex.
            if low_slack && pending.len()>=4 { std::thread::sleep(std::time::Duration::from_micros(10)); }
            else { std::thread::yield_now(); }
        }
    }
}
struct Launcher { tx: Option<mpsc::SyncSender<Command>>, join: Option<std::thread::JoinHandle<()>> }
impl Drop for Launcher {
    fn drop(&mut self) { self.tx.take(); if let Some(join) = self.join.take() { let _ = join.join(); } }
}
thread_local! { static LAUNCHER: std::cell::RefCell<Option<Launcher>> = const { std::cell::RefCell::new(None) }; }
pub fn submit(function: Arc<Function>, inputs: Buffers, outputs: Buffers, owners: Vec<Buffers>) -> Completion {
    let answer = Arc::new(Answer::default());
    let cmd = Command { function, inputs, outputs, owners, answer: answer.clone(), claimed: false };
    LAUNCHER.with_borrow_mut(|slot| {
        let launcher = slot.get_or_insert_with(|| {
            let (tx, rx) = mpsc::sync_channel(MAX_IN_FLIGHT);
            let join = std::thread::Builder::new().name("gemma4-waiter".into()).spawn(move || worker(rx)).expect("launch helper");
            eprintln!("ENGINE_LAUNCH_THREAD started bounded depth {}", MAX_IN_FLIGHT);
            Launcher { tx: Some(tx), join: Some(join) }
        });
        if launcher.tx.as_ref().unwrap().send(cmd).is_err() { answer.complete(Err(Error::Poisoned)); }
    });
    Completion(answer)
}

impl Drop for Command {
    fn drop(&mut self) {
        if !self.claimed { self.answer.complete(Err(Error::Poisoned)); }
    }
}
