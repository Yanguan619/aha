//! 运行时指标：`GET /metrics`（JSON）与 `GET /`（纯前端面板）。
//!
//! 计数全部在进程内，不落盘、不依赖外部服务；面板每秒轮询一次即可。

use axum::{
    Json,
    extract::{MatchedPath, Request},
    http::header,
    middleware::Next,
    response::{Html, IntoResponse, Response},
};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{ProcessesToUpdate, System};

/// 直方图桶上界（毫秒），最后一个槽位是 `+Inf`。
/// 取值贴着推理服务：5~50ms 是缓冲/JSON 小端点，100~2500ms 是 ASR/TTS，5s 以上是长生成。
const BUCKETS_MS: [f64; 17] = [
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 400.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0,
    5000.0, 8000.0, 12000.0, 30000.0,
];
const RECENT_CAP: usize = 100;
/// 1s 轮询下每 2s 采一次进程数据，采样本身（刷新全表）就不算负担。
const PROC_CACHE_TTL: Duration = Duration::from_secs(2);

static STATE: OnceLock<Mutex<MetricsState>> = OnceLock::new();
static PORT: OnceLock<u16> = OnceLock::new();
/// key（角色 / 模型 id）→ 展示名。加载后不会变，读一次即可。
static MODEL_NAMES: LazyLock<Mutex<HashMap<String, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// 记录监听端口（真实端口只有 `start_http_server` 知道）。
///
/// 顺手把状态建起来：uptime 要从服务启动算，不能从第一次请求算。
pub(crate) fn set_port(port: u16) {
    PORT.get_or_init(|| port);
    let _ = state().started;
    // sysinfo 首次初始化（WMI / 电源查询）实测 1.5s：在端口开之前后台付掉，
    // 否则面板第一次轮询要等一秒半。
    std::thread::spawn(|| {
        process_snapshot();
    });
}

#[derive(Clone, Default)]
struct RouteStat {
    requests: u64,
    errors: u64,
    inflight: i64,
    sum_ms: f64,
    max_ms: f64,
    last_ms: f64,
    last_at: f64,
    /// 每个桶的计数；`buckets[BUCKETS_MS.len()]` 是 `+Inf`。
    buckets: [u64; BUCKETS_MS.len() + 1],
}

impl RouteStat {
    fn observe(&mut self, ms: f64, failed: bool) {
        self.requests += 1;
        if failed {
            self.errors += 1;
        }
        self.sum_ms += ms;
        self.max_ms = self.max_ms.max(ms);
        self.last_ms = ms;
        self.last_at = now_secs();
        let idx = BUCKETS_MS
            .iter()
            .position(|&bound| ms <= bound)
            .unwrap_or(BUCKETS_MS.len());
        self.buckets[idx] += 1;
    }

    /// 分位数取所在桶的上界，不做桶内插值——桶够密，面板看趋势足够。
    fn quantile(&self, q: f64) -> f64 {
        let total: u64 = self.buckets.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let target = (q * total as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            seen += count;
            if seen >= target {
                return BUCKETS_MS.get(i).copied().unwrap_or(self.max_ms);
            }
        }
        self.max_ms
    }
}

#[derive(Default)]
struct Totals {
    requests: u64,
    errors: u64,
    inflight: i64,
}

#[derive(Clone)]
struct Recent {
    t: f64,
    method: String,
    path: String,
    status: u16,
    ms: f64,
}

struct MetricsState {
    started: Instant,
    totals: Totals,
    routes: HashMap<String, RouteStat>,
    recent: VecDeque<Recent>,
}

fn state() -> MutexGuard<'static, MetricsState> {
    STATE
        .get_or_init(|| {
            Mutex::new(MetricsState {
                started: Instant::now(),
                totals: Totals::default(),
                routes: HashMap::new(),
                recent: VecDeque::with_capacity(RECENT_CAP),
            })
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

/// 每个请求过一遍：路由标签 + 耗时 + 状态码，写进环形缓冲和直方图。
pub(crate) async fn track_metrics(request: Request, next: Next) -> Response {
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "(unmatched)".to_string());
    // 面板自己每秒打 /metrics 和 /，统计进来只会把 QPS 变成轮询频率。
    if route == "/metrics" || route == "/" {
        return next.run(request).await;
    }

    let method = request.method().as_str().to_string();
    let started = Instant::now();
    {
        let mut st = state();
        st.totals.inflight += 1;
        st.routes.entry(route.clone()).or_default().inflight += 1;
    }

    let response = next.run(request).await;

    let ms = started.elapsed().as_secs_f64() * 1000.0;
    let status = response.status().as_u16();
    {
        let mut st = state();
        st.totals.inflight -= 1;
        st.totals.requests += 1;
        if status >= 400 {
            st.totals.errors += 1;
        }
        let stat = st.routes.entry(route.clone()).or_default();
        stat.inflight -= 1;
        stat.observe(ms, status >= 400);
        st.recent.push_front(Recent {
            t: now_secs(),
            method,
            path: route,
            status,
            ms: round1(ms),
        });
        st.recent.truncate(RECENT_CAP);
    }
    response
}

/// `GET /metrics` — 面板轮询的 JSON。
pub(crate) async fn metrics_handler() -> Response {
    // 先拷一份再放锁：后面读模型注册表要 await，MutexGuard 不能跨 await。
    let (uptime, totals, mut routes, recent) = {
        let st = state();
        (
            st.started.elapsed().as_secs_f64(),
            (st.totals.requests, st.totals.errors, st.totals.inflight),
            st.routes
                .iter()
                .map(|(route, stat)| (route.clone(), stat.clone()))
                .collect::<Vec<_>>(),
            st.recent.iter().cloned().collect::<Vec<_>>(),
        )
    };
    routes.sort_by(|a, b| b.1.requests.cmp(&a.1.requests).then_with(|| a.0.cmp(&b.0)));

    // 模型名是启动时就定死的：读一次就缓存，且只用 try_read——推理会长时间持有写锁，
    // 监控接口不能被正在跑的那一次 TTS/ASR 堵住。
    let mut models = Vec::new();
    let mut cache = MODEL_NAMES.lock().unwrap_or_else(|e| e.into_inner());
    for key in crate::server::api::loaded_keys() {
        if !cache.contains_key(&key) {
            let Some(model) = crate::server::api::model_by_key(&key) else {
                continue;
            };
            let Ok(guard) = model.try_read() else {
                continue;
            };
            let display = guard.which_model.display_name();
            // 面板的字位只有一格：去掉 owner/ 前缀，sherpa 的长仓库名至少还能认出模型种类。
            cache.insert(
                key.clone(),
                display.rsplit('/').next().unwrap_or(&display).to_string(),
            );
        }
        if let Some(name) = cache.get(&key) {
            models.push(json!({ "key": key, "model": name }));
        }
    }
    drop(cache);

    let (mem_mb, cpu_pct) = process_snapshot();
    let body: Value = json!({
        "server": {
            "pid": std::process::id(),
            "port": PORT.get().copied().unwrap_or(0),
            "uptime_s": round1(uptime),
            "version": env!("CARGO_PKG_VERSION"),
        },
        "process": {
            "mem_mb": round1(f64::from(mem_mb)),
            "cpu_pct": round1(cpu_pct),
        },
        "models": models,
        "totals": {
            "requests": totals.0,
            "errors": totals.1,
            "inflight": totals.2,
        },
        "routes": routes
            .iter()
            .map(|(route, stat)| json!({
                "route": route,
                "requests": stat.requests,
                "errors": stat.errors,
                "inflight": stat.inflight,
                "avg_ms": round1(if stat.requests == 0 {
                    0.0
                } else {
                    stat.sum_ms / stat.requests as f64
                }),
                "p50_ms": round1(stat.quantile(0.50)),
                "p95_ms": round1(stat.quantile(0.95)),
                "p99_ms": round1(stat.quantile(0.99)),
                "max_ms": round1(stat.max_ms),
                "last_ms": round1(stat.last_ms),
                "last_at": stat.last_at,
            }))
            .collect::<Vec<Value>>(),
        "recent": recent
            .iter()
            .map(|r| json!({
                "t": r.t,
                "method": r.method,
                "path": r.path,
                "status": r.status,
                "ms": r.ms,
            }))
            .collect::<Vec<Value>>(),
    });

    ([(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")], Json(body)).into_response()
}

/// 进程提交内存（MB）：Windows 上取 `PagefileUsage`（≈ 任务管理器"提交大小"）。
#[cfg(windows)]
fn commit_mb(pid: u32) -> Option<f32> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        let ok = K32GetProcessMemoryInfo(
            handle,
            &mut counters,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        );
        CloseHandle(handle);
        (ok != 0).then(|| counters.PagefileUsage as f32 / (1024.0 * 1024.0))
    }
}

#[cfg(not(windows))]
fn commit_mb(_pid: u32) -> Option<f32> {
    None
}

/// `GET /` — 纯前端检测面板。
pub(crate) async fn panel() -> Html<&'static str> {
    Html(include_str!("ui/index.html"))
}

/// 本进程的提交内存（MB）与整机归一化的 CPU 百分比；查不到就是 0。
///
/// 内存**不用** sysinfo 的 `memory()`（= Windows 工作集）：aha 的模型权重大多记在
/// commit 里而不是常驻页，实测 WS 17MB / commit 2485MB——用 WS 会显示成"模型没加载"。
fn process_snapshot() -> (f32, f64) {
    static CACHE: Mutex<Option<(Instant, f32, f64)>> = Mutex::new(None);
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();

    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, mem_mb, cpu_pct)) = *cache {
        if at.elapsed() < PROC_CACHE_TTL {
            return (mem_mb, cpu_pct);
        }
    }

    // System 要留着复用：cpu_usage 是两次刷新之间的差值，新建的 System 首次永远是 0。
    let mut sys = SYS
        .get_or_init(|| Mutex::new(System::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    sys.refresh_cpu_usage();
    // 只刷新自己：`ProcessesToUpdate::All` 要枚举整机进程，推理满载时要花几秒，
    // 会把 /metrics 拖成秒级（实测 2.6s → 3ms）。
    let me = sysinfo::get_current_pid().ok();
    if let Some(pid) = me {
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    }
    let mem_mb = me
        .map(|pid| {
            commit_mb(pid.as_u32()).unwrap_or_else(|| {
                sys.process(pid)
                    .map(|proc_| proc_.memory() as f32 / (1024.0 * 1024.0))
                    .unwrap_or(0.0)
            })
        })
        .unwrap_or(0.0);
    let cpu_pct = me
        .and_then(|pid| sys.process(pid))
        .map(|proc_| {
            let cores = sys.cpus().len().max(1) as f64;
            f64::from(proc_.cpu_usage()) / cores
        })
        .unwrap_or(0.0);
    *cache = Some((Instant::now(), mem_mb, cpu_pct));
    (mem_mb, cpu_pct)
}
