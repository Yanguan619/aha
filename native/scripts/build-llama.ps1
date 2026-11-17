# 把 llama.cpp 编成 Android 静态库，放到 build.rs 期待的 llama-prebuilt/<abi>/。
# Windows 原生版（PowerShell + NDK 自带 CMake 工具链），和 build-llama.sh 等价。
#
# 用法：
#   $env:ANDROID_NDK_HOME="D:\Android\SDK\ndk\27.3.13750724"
#   powershell -File native\scripts\build-llama.ps1 -Abi arm64-v8a
#   cargo build --release --features llama --target aarch64-linux-android
#
# 源码要求同 build-llama.sh：带 STQ（PR #22836）的 llama.cpp。
# 没有 STQ 时会尝试应用 native/patches/llama-stq.patch；之后再打
# native/patches/llama-stq-legacy-id.patch（type id 42 -> STQ1_0）。

param(
    [string]$Abi = "arm64-v8a",
    [string]$LlamaSrc = $env:LLAMA_CPP_SRC,
    [string]$Ndk = $env:ANDROID_NDK_HOME,
    [int]$AndroidApi = 24
)

$ErrorActionPreference = "Stop"

switch ($Abi) {
    "arm64-v8a" { $androidAbi = "arm64-v8a" }
    "x86_64" { $androidAbi = "x86_64" }
    { $_ -in @("armeabi-v7a", "arm") } { $androidAbi = "armeabi-v7a" }
    default {
        Write-Error "不支持的 ABI: $Abi（可用：arm64-v8a / x86_64 / armeabi-v7a）"
        exit 1
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $LlamaSrc) { $LlamaSrc = (Join-Path $repoRoot "..\llama.cpp") }

# NDK 定位：显式参数 > 环境变量 > 常见 SDK 目录里版本号最大的 ndk。
function Find-Ndk {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-Path $Explicit)) { throw "NDK 路径不存在：$Explicit" }
        return (Resolve-Path $Explicit).Path
    }
    foreach ($env in @("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT")) {
        $v = [Environment]::GetEnvironmentVariable($env)
        if ($v -and (Test-Path $v)) { return $v }
    }
    $candidates = @()
    foreach ($sdk in @($env:ANDROID_HOME, $env:ANDROID_SDK_ROOT, "D:\Android\SDK", (Join-Path $env:LOCALAPPDATA "Android\Sdk"))) {
        if ($sdk -and (Test-Path (Join-Path $sdk "ndk"))) {
            $candidates += Get-ChildItem (Join-Path $sdk "ndk") -Directory
        }
    }
    if ($candidates.Count -eq 0) { throw "找不到 NDK：设置 ANDROID_NDK_HOME 或用 -Ndk 指定" }
    return ($candidates | Sort-Object Name -Descending | Select-Object -First 1).FullName
}

$Ndk = Find-Ndk $Ndk
$toolchain = Join-Path $Ndk "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $toolchain)) { throw "找不到 $toolchain，检查 NDK 路径" }
if (-not (Test-Path $LlamaSrc)) { throw "找不到 llama.cpp 源码：$LlamaSrc（用 -LlamaSrc 或 LLAMA_CPP_SRC 指定）" }

$patchFile = Join-Path $repoRoot "native\patches\llama-stq.patch"
$legacyPatch = Join-Path $repoRoot "native\patches\llama-stq-legacy-id.patch"

Write-Host "==> llama.cpp: $LlamaSrc"
Write-Host "==> NDK: $Ndk"
Write-Host "==> ABI: $androidAbi   API: $AndroidApi"

# STQ 是否已经在源码里：看 ggml 的量化类型定义。
$stqPresent = (Select-String -Path $LlamaSrc\ggml\src\ggml.c -Pattern "STQ1_0" -Quiet)
if (-not $stqPresent) {
    if (Test-Path $patchFile) {
        Write-Host "==> 源码不含 STQ，应用 $patchFile"
        git -C $LlamaSrc apply --whitespace=nowarn $patchFile
        if ($LASTEXITCODE -ne 0) {
            throw "补丁打不上：源码版本和 native/patches/llama-stq.patch 不一致。请改用带 PR #22836 的源码（git fetch origin pull/22836/head:stq）。"
        }
    } else {
        throw "源码里没有 STQ 量化，也没有 native/patches/llama-stq.patch"
    }
}

# 让 GGML_TYPE_Q2_0（id 42）按 STQ1_0 解码：腾讯的 1.25bit GGUF 用的还是旧编号。
$aliased = (Select-String -Path $LlamaSrc\ggml\src\ggml.c -Pattern "GGML_TYPE_Q2_0" -Context 0,4 |
    Where-Object { $_.Context.PostContext -match "block_stq1_0" -or $_.Line -match "block_stq1_0" })
if (-not $aliased) {
    if (Test-Path $legacyPatch) {
        Write-Host "==> 应用 type id 42 -> STQ1_0 别名补丁：$legacyPatch"
        git -C $LlamaSrc apply --whitespace=nowarn $legacyPatch
        if ($LASTEXITCODE -ne 0) { throw "legacy id 补丁打不上：$legacyPatch" }
    } else {
        Write-Warning "找不到 $legacyPatch，官方 1.25bit GGUF 加载时会报 tensor offset 不符"
    }
}

$buildDir = Join-Path $LlamaSrc "build-android-$androidAbi"
$targets = @("llama", "ggml", "ggml-cpu", "ggml-base")
cmake -S $LlamaSrc -B $buildDir -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
    -DANDROID_ABI="$androidAbi" `
    -DANDROID_PLATFORM="android-$AndroidApi" `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_SHARED_LIBS=OFF `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_SERVER=OFF `
    -DLLAMA_BUILD_TOOLS=OFF `
    -DGGML_NATIVE=OFF `
    -DGGML_OPENMP=OFF `
    -DGGML_LLAMAFILE=OFF `
    -DGGML_CCACHE=OFF `
    -DLLAMA_CURL=OFF
if ($LASTEXITCODE -ne 0) { throw "cmake 配置失败" }

cmake --build $buildDir --config Release -j 8 --target @targets
if ($LASTEXITCODE -ne 0) { throw "cmake 构建失败" }

$outDir = Join-Path $repoRoot "llama-prebuilt\$Abi"
New-Item -ItemType Directory -Force $outDir | Out-Null
$found = 0
foreach ($name in @("llama", "ggml", "ggml-cpu", "ggml-base", "ggml-blas")) {
    foreach ($lib in @(
            (Join-Path $buildDir "lib$name.a"),
            (Join-Path $buildDir "src\lib$name.a"),
            (Join-Path $buildDir "ggml\src\lib$name.a"))) {
        if (Test-Path $lib) {
            Copy-Item $lib $outDir -Force
            $found = 1
            break
        }
    }
}
if ($found -eq 0) { throw "构建完成但没找到 libllama.a / libggml*.a，检查 $buildDir" }

$cxxSo = Get-ChildItem (Join-Path $Ndk "toolchains\llvm\prebuilt") -Recurse -Filter "libc++_shared.so" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match [regex]::Escape($androidAbi) } |
    Select-Object -First 1
if ($cxxSo) {
    Copy-Item $cxxSo.FullName $outDir -Force
} else {
    Write-Warning "没找到 libc++_shared.so，链接阶段可能报缺 c++_shared"
}

$llvmNm = Get-ChildItem (Join-Path $Ndk "toolchains\llvm\prebuilt") -Recurse -Filter "llvm-nm.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($llvmNm) {
    $hit = & $llvmNm.FullName (Join-Path $outDir "libggml-cpu.a") 2>$null | Select-String -Pattern "stq1_0"
    if ($hit) {
        Write-Host "==> 静态库含 STQ 符号 ✓"
    } else {
        Write-Warning "libggml-cpu.a 里没看到 STQ 符号，Hy-MT2 1.25bit 可能仍加载失败"
    }
}

Write-Host "==> 完成：$outDir"
Get-ChildItem $outDir | Format-Table Name, Length -AutoSize
