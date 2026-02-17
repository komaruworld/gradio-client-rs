use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use serde_json::Value;
use tokio::sync::{broadcast, mpsc, Mutex, Notify};
use tokio_stream::wrappers::BroadcastStream;

use crate::error::{Error, Result};
use crate::types::{OutputUpdate, Status, StatusUpdate, Update};

#[derive(Debug)]
struct JobInner {
    latest_status: Mutex<StatusUpdate>,
    outputs: Mutex<Vec<Value>>,
    result: Mutex<Option<Result<Value>>>,
    done: AtomicBool,
    notify: Notify,
    updates_tx: broadcast::Sender<Update>,
    cancel_tx: mpsc::UnboundedSender<()>,
}

#[derive(Debug, Clone)]
pub struct Job {
    inner: Arc<JobInner>,
}

#[derive(Debug)]
pub(crate) struct JobWorkerHandle {
    inner: Arc<JobInner>,
    cancel_rx: mpsc::UnboundedReceiver<()>,
}

impl Job {
    pub(crate) fn new() -> (Self, JobWorkerHandle) {
        let (updates_tx, _) = broadcast::channel(512);
        let (cancel_tx, cancel_rx) = mpsc::unbounded_channel();
        let inner = Arc::new(JobInner {
            latest_status: Mutex::new(StatusUpdate::default()),
            outputs: Mutex::new(Vec::new()),
            result: Mutex::new(None),
            done: AtomicBool::new(false),
            notify: Notify::new(),
            updates_tx,
            cancel_tx,
        });

        (
            Self {
                inner: inner.clone(),
            },
            JobWorkerHandle { inner, cancel_rx },
        )
    }

    pub async fn result(&self) -> Result<Value> {
        loop {
            if let Some(result) = self.inner.result.lock().await.clone() {
                return result;
            }
            self.inner.notify.notified().await;
        }
    }

    pub async fn outputs(&self) -> Vec<Value> {
        self.inner.outputs.lock().await.clone()
    }

    pub async fn status(&self) -> StatusUpdate {
        let mut status = self.inner.latest_status.lock().await.clone();
        if self.done() && status.code != Status::Cancelled {
            if self
                .inner
                .result
                .lock()
                .await
                .as_ref()
                .is_some_and(Result::is_ok)
            {
                status.code = Status::Finished;
                status.success = Some(true);
            } else if self.inner.result.lock().await.is_some() {
                status.code = Status::Finished;
                status.success = Some(false);
            }
        }
        status
    }

    pub async fn cancel(&self) -> bool {
        let sent = self.inner.cancel_tx.send(()).is_ok();
        if sent {
            let mut status = self.inner.latest_status.lock().await;
            status.code = Status::Cancelled;
            status.success = Some(false);
            let _ = self.inner.updates_tx.send(Update::Status(status.clone()));
        }
        sent
    }

    pub fn done(&self) -> bool {
        self.inner.done.load(Ordering::SeqCst)
    }

    pub fn updates(&self) -> impl Stream<Item = Update> {
        BroadcastStream::new(self.inner.updates_tx.subscribe())
            .filter_map(|item| async move { item.ok() })
    }
}

impl JobWorkerHandle {
    pub async fn recv_cancel(&mut self) -> Option<()> {
        self.cancel_rx.recv().await
    }

    pub fn try_recv_cancel(&mut self) -> bool {
        self.cancel_rx.try_recv().is_ok()
    }

    pub async fn set_status(&self, status: StatusUpdate) {
        {
            let mut guard = self.inner.latest_status.lock().await;
            *guard = status.clone();
        }
        let _ = self.inner.updates_tx.send(Update::Status(status));
    }

    pub async fn push_output(&self, output: Value, success: bool, final_output: bool) {
        {
            let mut outputs = self.inner.outputs.lock().await;
            outputs.push(output.clone());
        }
        let _ = self.inner.updates_tx.send(Update::Output(OutputUpdate {
            outputs: output,
            success,
            final_output,
        }));
    }

    pub async fn finish(&self, result: Result<Value>) {
        {
            let mut guard = self.inner.result.lock().await;
            *guard = Some(result);
        }
        self.inner.done.store(true, Ordering::SeqCst);
        self.inner.notify.notify_waiters();
    }

    pub async fn has_outputs(&self) -> bool {
        !self.inner.outputs.lock().await.is_empty()
    }

    pub async fn mark_cancelled(&self) {
        let mut status = self.inner.latest_status.lock().await;
        status.code = Status::Cancelled;
        status.success = Some(false);
        let _ = self.inner.updates_tx.send(Update::Status(status.clone()));
        self.inner.done.store(true, Ordering::SeqCst);
        {
            let mut guard = self.inner.result.lock().await;
            *guard = Some(Err(Error::Cancelled));
        }
        self.inner.notify.notify_waiters();
    }
}
