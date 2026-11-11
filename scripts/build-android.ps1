# 交叉编译 aha 到 Android（aarch64-linux-android）。用法：
#   .\scripts\build-android.ps1                 # debug
#   .\scripts\build-android.ps1 -Release        # release
#   .\scripts\build-android.ps1 -Release -Features sherpa,ffmpeg
# NDK 未指定时依次找 ANDROID_NDK_HOME / ANDROID_NDK_ROOT / ANDROID_HOME\ndk / LOCALAPPDATA\Android\Sdk\ndk。
# 注意：本文件必须存成 UTF-8 **带 BOM**，否则 Windows PowerShell 5.1 会按 GBK 解析中文注释而报语法错误。
param(
    [string]$Ndk,
    [int]$Api = 26,
    [switch]$Release,
    [string]$Features
)

$ErrorActionPreference = "Stop"
$triple = "aarch64-linux-android"

# ---- 1. NDK ----
if (-not $Ndk) {
    $cands = @($env:ANDROID_NDK_HOME, $env:ANDROID_NDK_ROOT) | Where-Object { $_ -and (Test-Path $_) }
    foreach ($sdk in @($env:ANDROID_HOME, $env:ANDROID_SDK_ROOT, "$env:LOCALAPPDATA\Android\Sdk", "D:\Android\Sdk")) {
        if ($sdk -and (Test-Path "$sdk\ndk")) { $cands += (Get-ChildItem "$sdk\ndk" -Directory | Sort-Object Name | Select-Object -Last 1).FullName }
    }
    $Ndk = $cands | Select-Object -First 1
}
if (-not $Ndk -or -not (Test-Path $Ndk)) { throw "找不到 NDK，请用 -Ndk <路径> 指定（例如 D:\Android\Sdk\ndk\26.3.11579264）" }

$llvm = Join-Path $Ndk "toolchains\llvm\prebuilt\windows-x86_64\bin"
$ndkSysroot = Join-Path $Ndk "toolchains\llvm\prebuilt\windows-x86_64\sysroot"
# 用 clang.exe 而不是 aarch64-linux-androidNN-clang.cmd：后者是批处理，rustc/cc 传过来的
# 参数里有 `-Wl,--version-script=...` 这种带引号的，cmd 会按自己的规则重新解析并报
# “此时不应有 --version-script=...”（cdylib 的 crc-fast 就是这样挂的）。
# target/sysroot 改成显式 flag 传给裸 clang。
$target = "aarch64-linux-android$Api"
$clang = Join-Path $llvm "clang.exe"
$clangxx = Join-Path $llvm "clang++.exe"
if (-not (Test-Path $clang)) { throw "NDK 里没有 clang.exe: $llvm" }
$clangFlags = "--target=$target --sysroot=$ndkSysroot"
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw "缺少 ninja（CMake 在 Windows 上默认用 VS 生成器，无法交叉编译）：pip install ninja" }

# ---- 2. rust-std for the target ----
$sysroot = (rustc --print sysroot)
$stdDir = Join-Path $sysroot "lib\rustlib\$triple"
if (-not (Test-Path $stdDir)) {
    "==> 安装 rust-std ($triple)"
    $ok = $false
    try {
        rustup target add $triple 2>&1 | Out-Null
        $ok = Test-Path (rustc --target $triple --print target-libdir 2>$null)
    } catch {}
    if (-not $ok) {
        # rustup 走镜像时经常 404（镜像缺该 target 的 rust-std），直接从 rsproxy 拉同一份包手动落盘。
        $manifest = Join-Path $sysroot "lib\rustlib\multirust-channel-manifest.toml"
        $date = (Select-String -Path $manifest -Pattern '^date = "(.+)"' -List).Matches.Groups[1].Value
        $ver = ((rustc -V) -split ' ')[1]
        $tmp = Join-Path $env:TEMP "aha-rust-std-$triple"
        New-Item -ItemType Directory -Force $tmp | Out-Null
        $url = "https://rsproxy.cn/dist/$date/rust-std-$ver-$triple.tar.xz"
        "==> rustup 不可用，改从 $url 下载"
        curl.exe -sL --fail --max-time 900 -o "$tmp\rust-std.tar.xz" $url
        tar -xf "$tmp\rust-std.tar.xz" -C $tmp
        if ($LASTEXITCODE -ne 0) { throw "下载或解压 rust-std 失败：$url" }
        Copy-Item (Join-Path $tmp "rust-std-$ver-$triple\rust-std-$triple\lib\rustlib\$triple") $stdDir -Recurse -Force
    }
}

# ---- 3. CMake：用 CMake 自带的 Android 支持，不用 NDK 的 legacy toolchain 文件 ----
# legacy 文件默认 ABI 是 armeabi-v7a，会跟 clang 的 --target=aarch64 打架。
$cmakeToolchain = Join-Path $env:TEMP "aha-android-toolchain.cmake"
@"
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION $Api)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_NDK "$($Ndk -replace '\\','/')")
set(CMAKE_ANDROID_STL_TYPE c++_static)
"@ | Set-Content $cmakeToolchain -Encoding ascii

# 上一次用 VS 生成器留下的缓存会让 cmake 直接报 "generator does not match"。
Get-ChildItem (Join-Path $PSScriptRoot "..\target\$triple") -Recurse -Filter CMakeCache.txt -ErrorAction SilentlyContinue |
    Where-Object { Select-String -Path $_.FullName -Pattern 'CMAKE_GENERATOR:INTERNAL=Visual Studio' -Quiet } |
    ForEach-Object { "==> 清理旧 VS 生成器缓存: $($_.Directory.Parent.FullName)"; Remove-Item $_.Directory.Parent.FullName -Recurse -Force }

# ---- 4. build ----
$env:PATH = "$llvm;$env:PATH"
# 裸 clang 不带默认 target，CC/CXX/链接都要显式给 --target 和 --sysroot。
$env:CC_aarch64_linux_android = $clang
$env:CXX_aarch64_linux_android = $clangxx
$env:CFLAGS_aarch64_linux_android = $clangFlags
$env:CXXFLAGS_aarch64_linux_android = $clangFlags
$env:AR_aarch64_linux_android = Join-Path $llvm "llvm-ar.exe"
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = $clangxx
# 环境变量里的 rustflags 会覆盖 .cargo/config.toml 的同 target 配置，所以 +fp16 也带在这。
# gemm-f16 的内联汇编要 fullfp16；两个 link-arg 补上裸 clang 缺的 target/sysroot。
# max-page-size：Android 15+ 要求 .so 的 LOAD 段按 16KB 对齐（默认 4KB，装到手机上会弹
# "不符合 16 KB 对齐要求"）。NDK 27 默认就是 16KB，这里显式写一遍防退化。
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS =
    "-C target-feature=+fp16 -C link-arg=--target=$target -C link-arg=--sysroot=$ndkSysroot " +
    "-C link-arg=-Wl,-z,max-page-size=16384"
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_TOOLCHAIN_FILE = $cmakeToolchain

$cargoArgs = @("build", "--target", $triple)
if ($Release) { $cargoArgs += "--release" }
if ($Features) { $cargoArgs += @("--features", $Features) }
"==> cargo $($cargoArgs -join ' ')"
# 原生程序的 stderr 在 PS 5.1 里会变成 error record；开着 Stop 时调用方一重定向就会中断脚本。
$ErrorActionPreference = "Continue"
& cargo @cargoArgs
$cargoExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($cargoExit -ne 0) { throw "cargo build 失败（exit $cargoExit）" }

$profile = if ($Release) { "release" } else { "debug" }
$out = Join-Path $PSScriptRoot "..\target\$triple\$profile\aha"
"==> 产物: $((Resolve-Path $out).Path) ($([math]::Round((Get-Item $out).Length / 1MB)) MB)"
"    依赖: " + ((& (Join-Path $llvm "llvm-readelf.exe") -d $out 2>$null | Select-String 'NEEDED') -join ', ').Trim()
