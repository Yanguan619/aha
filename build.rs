//! Build glue for the optional llama.cpp backend.
//!
//! The llama.cpp static libraries are produced out-of-band by `native/scripts/build-llama.sh`
//! (one directory per Android ABI under `native/llama-prebuilt/<abi>/`), because llama.cpp is a
//! CMake project and we do not want to drive CMake from inside `cargo`.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LLAMA");
    if std::env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "android" {
        // Host builds do not link llama.cpp: it is an on-device backend.
        return;
    }

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let abi = match arch.as_str() {
        "aarch64" => "arm64-v8a",
        "x86_64" => "x86_64",
        "arm" => "armeabi-v7a",
        other => panic!("unsupported android arch for llama.cpp backend: {other}"),
    };

    let dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("llama-prebuilt")
        .join(abi);

    if !dir.exists() {
        panic!(
            "llama.cpp static libraries not found at {}.\n\
             Run native/scripts/build-llama.sh {} first, or build without --features llama.",
            dir.display(),
            abi
        );
    }

    println!("cargo:rustc-link-search=native={}", dir.display());
    // Link order matters: llama depends on ggml, ggml depends on ggml-cpu / ggml-base.
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=dylib=c++_shared");
    println!("cargo:rustc-link-lib=dylib=log");
    // Rust links Android binaries with `-nodefaultlibs`, and the llama.cpp static libraries
    // reference bionic directly (`stderr`, `posix_madvise`, fortify `__readlink_chk`, ...), so the
    // C runtime has to be named explicitly or the bin link fails with undefined symbols.
    println!("cargo:rustc-link-lib=dylib=c");
    println!("cargo:rerun-if-changed={}", dir.display());
}
