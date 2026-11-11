//! 统一的 HTTPS 客户端：用 webpki-roots 内置根证书，不依赖系统证书库。
//!
//! reqwest 0.13 的 `rustls` feature 默认走 `rustls-platform-verifier`，Android 上
//! 要求宿主先初始化 JVM context（`Expect rustls-platform-verifier to be initialized`），
//! 纯 Rust 进程直接 panic。这里换成自带根证书的 rustls 配置，桌面和 Android 通用。

use anyhow::Result;

fn tls_config() -> rustls::ClientConfig {
    let roots = rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth()
}

/// 异步客户端（模型下载等）。
pub fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .use_preconfigured_tls(tls_config())
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?)
}

/// 阻塞客户端（sherpa 权重下载等同步路径）。
/// 30 分钟是"整条请求"的上限：model.onnx（325MB）能在 180KB/s 的线下完。
pub fn blocking_client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .use_preconfigured_tls(tls_config())
        .timeout(std::time::Duration::from_secs(1800))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?)
}
