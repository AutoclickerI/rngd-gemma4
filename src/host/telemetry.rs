//! Read-only RNGD sensors. All discovery, reads and logging run on a
//! detached observer thread.
//! Missing/inaccessible/malformed sensors are null, never zero or a serving error.
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const PERIOD: Duration = Duration::from_millis(100);
const WINDOW: Duration = Duration::from_secs(10);
static START: Once = Once::new();
/// Spawn or sensor failures never fail the server.
pub fn start() {
    START.call_once(|| {
        if let Err(error) = std::thread::Builder::new().name("npu-telemetry".into()).spawn(|| observe(Path::new("/sys"))) {
            let _ = writeln!(std::io::stderr(), "NPU_TELEMETRY unavailable: thread spawn: {error}");
        }
    });
}

#[derive(Debug)]
struct Card {
    index: u32,
    mgmt: PathBuf,
    hwmon: Option<PathBuf>,
    bus: Option<String>,
    history: VecDeque<(Instant, Sample)>,
}

fn read(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
}

fn discover(root: &Path) -> Vec<Card> {
    let mut cards = Vec::new();
    let Ok(entries) = fs::read_dir(root.join("class/rngd_mgmt")) else { return cards };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(index) = name.to_str().and_then(|n| n.strip_prefix("rngd!npu"))
            .and_then(|n| n.strip_suffix("mgmt")).and_then(|n| n.parse::<u32>().ok()) else { continue };
        let mgmt = entry.path();
        // hwmon indices need not equal NPU indices. Bind by the driver name.
        let hwmon = fs::read_dir(root.join("class/hwmon")).ok().and_then(|entries| {
            entries.flatten().map(|e| e.path()).find(|p| read(&p.join("name")).as_deref() == Some(&format!("rngd{index}")))
        });
        let bus = read(&mgmt.join("busname"));
        cards.push(Card { index, mgmt, hwmon, bus, history: VecDeque::new() });
    }
    cards.sort_by_key(|c| c.index);
    cards
}

fn clocks(text: &str) -> Option<Vec<u64>> {
    let values: Vec<u64> = text.split_whitespace().map(str::parse).collect::<Result<_, _>>().ok()?;
    (!values.is_empty() && values.iter().all(|&n| n > 0)).then_some(values)
}

fn reason(text: &str) -> Option<u32> {
    let value = text.strip_prefix("Throttle reason =").unwrap_or(text).trim();
    match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        Some(value) => u32::from_str_radix(value, 16).ok(),
        None => value.parse().ok(),
    }
}

#[derive(Clone, Debug)]
struct Sample {
    clocks: Option<Vec<u64>>,
    power_uw: Option<u64>,
    temp_mc: Option<i64>,
    throttle: Option<u32>,
}

impl Card {
    fn sample(&self) -> Sample {
        let scalar = |name: &str| self.hwmon.as_ref().and_then(|p| read(&p.join(name)));
        Sample {
            clocks: read(&self.mgmt.join("npu_clocks")).and_then(|s| clocks(&s)),
            power_uw: scalar("power1_average").and_then(|s| s.parse().ok()),
            // RNGD's temp1 is the PEAK sensor (millidegrees C).
            temp_mc: scalar("temp1_input").and_then(|s| s.parse().ok()),
            throttle: read(&self.mgmt.join("throttle_reason")).and_then(|s| reason(&s)),
        }
    }

    fn remember(&mut self, now: Instant, sample: Sample) {
        while self.history.front().is_some_and(|(t, _)| now.duration_since(*t) >= WINDOW) {
            self.history.pop_front();
        }
        // Bound storage even if the observation period changes in future.
        if self.history.len() == 101 { self.history.pop_front(); }
        self.history.push_back((now, sample));
    }

    fn summary(&self) -> String {
        let valid_clocks: Vec<_> = self.history.iter().filter_map(|(_, s)| s.clocks.as_ref()).collect();
        let min_hz = valid_clocks.iter().flat_map(|v| v.iter()).copied().min();
        let power: Vec<_> = self.history.iter().filter_map(|(_, s)| s.power_uw).collect();
        let mean_power = (!power.is_empty()).then(|| power.iter().map(|&p| p as f64).sum::<f64>() / power.len() as f64);
        let max_temp = self.history.iter().filter_map(|(_, s)| s.temp_mc).max();
        let reasons: Vec<_> = self.history.iter().filter_map(|(_, s)| s.throttle).collect();
        let mut text = format!("\"samples\":{},\"clock_samples\":{},\"power_samples\":{},\"temp_samples\":{},\"throttle_samples\":{},\"min_hz\":{},\"mean_power_uw\":{},\"max_temp_mc\":{}",
            self.history.len(), valid_clocks.len(), power.len(), self.history.iter().filter(|(_, s)| s.temp_mc.is_some()).count(), reasons.len(), number(min_hz), number(mean_power), number(max_temp));
        for (label, bit) in [("thermal", 2), ("power_cap", 4), ("clock_cap", 8)] {
            let _ = write!(text, ",\"{label}_samples\":{}", reasons.iter().filter(|&&r| r & bit != 0).count());
        }
        text
    }
}

fn number<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
}

fn quoted(value: Option<&str>) -> String {
    match value {
        None => "null".into(),
        Some(s) => {
            let mut out = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    c if c.is_control() => { let _ = write!(out, "\\u{:04x}", c as u32); }
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
    }
}

fn observe(root: &Path) {
    let started = Instant::now();
    let mut cards = discover(root);
    if cards.is_empty() {
        let _ = writeln!(std::io::stderr(), "NPU_TELEMETRY {{\"event\":\"unavailable\",\"reason\":\"no accessible rngd management nodes\"}}");
        return;
    }
    let mut last_summary = started;
    loop {
        let tick = Instant::now();
        let unix_ms = SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_millis());
        let summary = tick.duration_since(last_summary) >= Duration::from_secs(1);
        let mut batch = String::new();
        for card in &mut cards {
            let before = Instant::now();
            let sample = card.sample();
            let read_us = before.elapsed().as_micros();
            let hz = sample.clocks.as_ref().map(|v| format!("{v:?}")).unwrap_or_else(|| "null".into());
            let _ = writeln!(batch, "NPU_TELEMETRY {{\"event\":\"sample\",\"unix_ms\":{},\"elapsed_ms\":{:.3},\"npu\":{},\"bus\":{},\"clocks_hz\":{},\"power_uw\":{},\"temp_mc\":{},\"throttle_reason\":{},\"read_us\":{}}}",
                number(unix_ms), tick.duration_since(started).as_secs_f64()*1000., card.index, quoted(card.bus.as_deref()), hz, number(sample.power_uw), number(sample.temp_mc), number(sample.throttle), read_us);
            card.remember(tick, sample);
            if summary {
                let _ = writeln!(batch, "NPU_TELEMETRY {{\"event\":\"rolling\",\"unix_ms\":{},\"npu\":{},\"window_ms\":10000,{}}}", number(unix_ms), card.index, card.summary());
            }
        }
        // One best-effort write per sample batch. No panic if stderr is closed.
        if std::io::stderr().write_all(batch.as_bytes()).is_err() { return; }
        if summary { last_summary = tick; }
        std::thread::sleep(PERIOD.saturating_sub(tick.elapsed()));
    }
}
