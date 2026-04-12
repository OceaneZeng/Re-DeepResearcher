<#
Run script for Qwen2.5-7B-Instruct SFT using recipes/Qwen2.5-7B-Instruct/sft/config_local.yaml

Usage examples:
# Run directly with python (no accelerate):
PS> .\scripts\run_qwen2.5_sft.ps1

# Run with accelerate (uses recipes/accelerate_configs/fsdp.yaml by default):
PS> .\scripts\run_qwen2.5_sft.ps1 -UseAccelerate

# Pass extra args to sft.py (they are forwarded):
PS> .\scripts\run_qwen2.5_sft.ps1 -- --per_device_train_batch_size 1 --num_train_epochs 1

Notes:
- This script expects to be executed from the repository root (where this script lives under ./scripts).
- The default config file is recipes/Qwen2.5-7B-Instruct/sft/config_local.yaml. You can override it with -ConfigFile.
- If you want to use vllm/bitsandbytes on Windows, please prefer WSL / Linux. The local config sets use_liger_kernel=false by default.
#>

param(
    [string]$ConfigFile = "recipes/Qwen2.5-7B-Instruct/sft/config_local.yaml",
    [string]$AccelerateConfig = "recipes/accelerate_configs/fsdp.yaml",
    [switch]$UseAccelerate,
    [switch]$ShowHelp,
    [switch]$ForceCliFallback = $true
)

if ($ShowHelp) {
    Write-Host "Usage: .\scripts\run_qwen2.5_sft.ps1 [-ConfigFile <path>] [-UseAccelerate] [-ForceCliFallback:$false] [-- <extra sft.py args>]"
    exit 0
}

# Resolve repository root (assume script lives in <repo>/scripts)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

# Ensure the repository `src` directory is on PYTHONPATH so local package imports (e.g. `import open_r1`) work
# This avoids ModuleNotFoundError: No module named 'open_r1' when running without installing the package.
$SrcPath = Join-Path $RepoRoot.Path "src"
if ($env:PYTHONPATH) {
    # Prepend src only if it isn't already present to avoid duplicate entries on repeated runs
    $paths = $env:PYTHONPATH -split ';' | Where-Object { $_ -and ($_ -ne $SrcPath) }
    $env:PYTHONPATH = @($SrcPath) + $paths -join ';'
} else {
    $env:PYTHONPATH = $SrcPath
}
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"

# On Windows Python may use the system ANSI code page (e.g., CP936/GBK)
# which can cause `open(...).read()` to raise UnicodeDecodeError for files encoded in UTF-8.
# Enable Python's UTF-8 mode so that `pathlib.Path.read_text()` and `open()` default to UTF-8.
$env:PYTHONUTF8 = "1"
$env:LANG = "en_US.UTF-8"
Write-Host "Enabled Python UTF-8 mode: PYTHONUTF8=$env:PYTHONUTF8, LANG=$env:LANG"

# Reduce CUDA allocator fragmentation on smaller GPUs.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Write-Host "Set PYTORCH_CUDA_ALLOC_CONF=$env:PYTORCH_CUDA_ALLOC_CONF"

# Resolve config file path (allow both relative and absolute)
$FullConfig = $ConfigFile
if (-not (Test-Path $FullConfig)) {
    # try relative to repo root
    $FullConfig = Join-Path $RepoRoot.Path $ConfigFile
}

if (-not (Test-Path $FullConfig)) {
    Write-Error "Config file not found: $ConfigFile (tried: $FullConfig). Please provide a valid path using -ConfigFile.";
    exit 2
}

# Forward any extra arguments supplied after -- to the python script
$forwardedArgs = @()
if ($args.Count -gt 0) {
    $forwardedArgs += $args
}

# Helpers to run the command
function Run-WithPython($configPath, $extraArgs) {
    # Wrap in cmd to set PYTHONUTF8 in the child process and run python with -X utf8
    $cmd = "set PYTHONUTF8=1 && python -X utf8 src/open_r1/sft.py --config_file $configPath"
    if ($extraArgs) { $cmd += " " + ($extraArgs -join ' ') }

    Write-Host "Running (cmd wrapper): cmd /c $cmd"
    & cmd /c $cmd
    return $LASTEXITCODE
}

function Run-WithCliFallback($extraArgs) {
    $cmd = 'set PYTHONUTF8=1 && python -X utf8 src/open_r1/sft.py --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name open-r1/Mixture-of-Thoughts --dataset_config all --eos_token "<|im_end|>" --output_dir model/Qwen2.5-7B-Instruct-sft'
    if ($extraArgs) { $cmd += " " + ($extraArgs -join ' ') }

    Write-Host "Running fallback (cmd wrapper): cmd /c $cmd"
    & cmd /c $cmd
    return $LASTEXITCODE
}

function Run-WithAccelerate($accelConfig, $configPath, $extraArgs) {
    # Some tools spawned by accelerate may still pick up the system encoding; wrap with cmd to set env var explicitly
    $accelCmd = "set PYTHONUTF8=1 && accelerate launch --config_file=$accelConfig src/open_r1/sft.py --config_file $configPath"
    if ($extraArgs) { $accelCmd += " " + ($extraArgs -join ' ') }

    Write-Host "Running (cmd wrapper): cmd /c $accelCmd"
    & cmd /c $accelCmd
    return $LASTEXITCODE
}

# Check for accelerate availability if requested
if ($UseAccelerate) {
    try {
        $which = Get-Command accelerate -ErrorAction Stop
    } catch {
        Write-Warning "accelerate not found in PATH. Falling back to direct python run. Install accelerate or run this script without -UseAccelerate."
        $UseAccelerate = $false
    }
}

if ($ForceCliFallback) {
    $rc = Run-WithCliFallback $forwardedArgs
} elseif ($UseAccelerate) {
    $rc = Run-WithAccelerate $AccelerateConfig $FullConfig $forwardedArgs
} else {
    $rc = Run-WithPython $FullConfig $forwardedArgs
}

exit $rc
