# Verify the per-vendor runtime path on this machine and print a Markdown
# checklist ready to paste into an issue. Run after the store has finished
# "Set up the runtime" and, ideally, after each service has been started once
# from its screen. Read-only: it never installs, starts or stops anything.
#
#   powershell -ExecutionPolicy Bypass -File scripts\verify-vendor.ps1
#
# Optional: -Expect nvidia|amd|intel|cpu marks the vendor line as a failure
# when the store's own probe disagrees.
param([string]$Expect = "")

$ErrorActionPreference = "SilentlyContinue"
$distro = "ai-app-store"
$results = New-Object System.Collections.Generic.List[object]

function Add-Check($ok, $label, $detail) {
  $results.Add([pscustomobject]@{ ok = $ok; label = $label; detail = $detail })
}

function Wsl($script) {
  $out = & wsl.exe -d $distro -u root --exec bash -c $script 2>$null
  if ($out -is [array]) { return ($out -join "`n") } else { return "$out" }
}

function Http($url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
    return $r.StatusCode
  } catch { return 0 }
}

# ---------------------------------------------------------------------------
# Hardware: same registry key and PnP class the store's probe reads.
# ---------------------------------------------------------------------------
$gpus = Get-CimInstance Win32_VideoController | ForEach-Object {
  $vendor = switch -Regex ($_.PNPDeviceID) {
    "VEN_10DE" { "nvidia" }
    "VEN_1002" { "amd" }
    "VEN_8086" { "intel" }
    default { $null }
  }
  if ($vendor) { [pscustomobject]@{ name = $_.Name; vendor = $vendor; driver = $_.DriverVersion } }
}
$npus = Get-PnpDevice -PresentOnly | Where-Object {
  $_.Class -eq "ComputeAccelerator" -or $_.FriendlyName -cmatch "\bAI Boost\b|\bNPU\b|\bIPU\b|Hexagon"
} | Select-Object -ExpandProperty FriendlyName

$hasDiscreteNvidia = $gpus | Where-Object vendor -eq "nvidia"
$primaryVendor = if ($hasDiscreteNvidia) { "nvidia" } elseif ($gpus | Where-Object vendor -eq "amd") { "amd" } elseif ($gpus | Where-Object vendor -eq "intel") { "intel" } else { "cpu" }
$forced = $env:AIAS_VENDOR
$effective = if ($forced) { $forced.ToLower() } else { $primaryVendor }

$gpuList = ($gpus | ForEach-Object { "$($_.name) [$($_.vendor)] driver $($_.driver)" }) -join "; "
Add-Check ($gpus.Count -gt 0 -or $effective -eq "cpu") "GPUs detected" $gpuList
Add-Check $true "NPUs detected" $(if ($npus) { $npus -join "; " } else { "none" })
$vendorOk = if ($Expect) { $effective -eq $Expect.ToLower() } else { $true }
Add-Check $vendorOk "Vendor the runtime is built around" ("{0}{1}" -f $effective, $(if ($forced) { " (forced via AIAS_VENDOR=$forced)" } else { "" }))

# ---------------------------------------------------------------------------
# WSL2 distro and Docker
# ---------------------------------------------------------------------------
$distros = (& wsl.exe -l -q 2>$null) -replace "`0", "" | ForEach-Object { $_.Trim() }
Add-Check ($distros -contains $distro) "Distro `"$distro`" registered" ($distros -join ", ")
$dockerActive = (Wsl "systemctl is-active docker").Trim()
Add-Check ($dockerActive -eq "active") "Docker Engine active in the distro" $dockerActive
$nvidiaRuntime = Wsl "docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && echo yes || echo no"
if ($effective -eq "nvidia") {
  Add-Check ($nvidiaRuntime.Trim() -eq "yes") "NVIDIA container runtime registered" $nvidiaRuntime.Trim()
} else {
  Add-Check $true "NVIDIA container runtime" "not needed for $effective (containers run on the CPU)"
}

# ---------------------------------------------------------------------------
# Ollama: WSL on NVIDIA, native Windows process everywhere else.
# ---------------------------------------------------------------------------
$wslOllamaEnabled = (Wsl "e=`$(systemctl is-enabled ollama 2>/dev/null); echo `${e:-absent}").Trim()
$listeners = Get-NetTCPConnection -State Listen -LocalPort 11434 | Select-Object -ExpandProperty LocalAddress -Unique
$nativeExe = @(
  "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
  "$env:LOCALAPPDATA\ai-app-store\ollama\ollama.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
$nativeProc = Get-Process ollama -ErrorAction SilentlyContinue | Select-Object -First 1
# WSL publishes on [::1] only, a native serve binds 127.0.0.1; "localhost" tries both.
$version = try { (Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 3).version } catch { $null }

if ($effective -eq "nvidia") {
  Add-Check ($wslOllamaEnabled -eq "enabled") "Ollama enabled inside the distro (CUDA)" $wslOllamaEnabled
  Add-Check (-not $nativeProc) "No native ollama.exe competing for :11434" $(if ($nativeProc) { "ollama.exe pid $($nativeProc.Id) is running" } else { "none running" })
} else {
  Add-Check ($wslOllamaEnabled -ne "enabled") "WSL Ollama disabled (not installed or disabled)" $wslOllamaEnabled
  Add-Check ([bool]$nativeExe) "Native ollama.exe present" $(if ($nativeExe) { $nativeExe } else { "missing" })
  Add-Check ([bool]$nativeProc) "Native ollama serve running" $(if ($nativeProc) { "pid $($nativeProc.Id)" } else { "not running" })
  Add-Check ($listeners -contains "127.0.0.1") "Native Ollama bound to 127.0.0.1:11434" ($listeners -join ", ")
}
Add-Check ([bool]$version) "Ollama answers on localhost:11434" $(if ($version) { "version $version" } else { "no answer" })
if ($nativeProc -and $effective -ne "nvidia") {
  $backend = try { (Invoke-RestMethod http://127.0.0.1:11434/api/ps -TimeoutSec 3).models | ForEach-Object { "$($_.name): $($_.size_vram) bytes in VRAM" } } catch { $null }
  Add-Check $true "Loaded models and VRAM use (run a chat first)" $(if ($backend) { $backend -join "; " } else { "no model loaded" })
}

# ---------------------------------------------------------------------------
# Platform services: image flavour and health per vendor.
# ---------------------------------------------------------------------------
$services = @(
  @{ name = "aias-svc-speaches"; label = "Speaches"; port = 8880; health = "/v1/models"; gpuImage = "latest-cuda"; cpuImage = "latest-cpu" },
  @{ name = "aias-svc-comfyui"; label = "ComfyUI"; port = 8188; health = "/system_stats"; gpuImage = "cu126"; cpuImage = ":cpu" },
  @{ name = "aias-svc-cv"; label = "CV server"; port = 8900; health = "/v2/health/ready"; gpuImage = "tritonserver"; cpuImage = "model_server" }
)
foreach ($svc in $services) {
  $image = (Wsl "docker inspect -f '{{.Config.Image}}' $($svc.name) 2>/dev/null").Trim()
  if (-not $image) {
    Add-Check $true "$($svc.label) container" "not started yet (start it from its screen, then rerun)"
    continue
  }
  $want = if ($effective -eq "nvidia") { $svc.gpuImage } else { $svc.cpuImage }
  Add-Check ($image -like "*$want*") "$($svc.label) uses the $effective flavour" $image
  $code = Http "http://localhost:$($svc.port)$($svc.health)"
  Add-Check ($code -ge 200 -and $code -lt 400) "$($svc.label) healthy on localhost:$($svc.port)" "HTTP $code"
}

# ---------------------------------------------------------------------------
# Native Windows services must listen on loopback only (issue #22). A listener
# on 0.0.0.0 makes Windows Defender Firewall raise its "allow this app" dialog
# the first time the executable binds, which blocks unattended provisioning.
# WSL-published ports arrive through wslrelay / wslhost and are not native.
# ---------------------------------------------------------------------------
$nativeNames = @("ollama", "ovms", "whisper-server", "python")
$nativeListeners = Get-NetTCPConnection -State Listen -LocalPort 11434, 8188, 8881, 8900 | ForEach-Object {
  $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
  if ($proc -and ($nativeNames -contains $proc.ProcessName)) {
    [pscustomobject]@{ proc = $proc.ProcessName; addr = $_.LocalAddress; port = $_.LocalPort }
  }
}
if ($nativeListeners) {
  $bad = @($nativeListeners | Where-Object { $_.addr -ne "127.0.0.1" -and $_.addr -ne "::1" })
  $listing = ($nativeListeners | ForEach-Object { "$($_.proc) $($_.addr):$($_.port)" }) -join "; "
  Add-Check ($bad.Count -eq 0) "Native services listen on loopback only" $listing
} else {
  Add-Check $true "Native services listen on loopback only" "no native service running (start one, then rerun)"
}

# ---------------------------------------------------------------------------
# Local Space builds: CPU images on non-NVIDIA vendors.
# ---------------------------------------------------------------------------
$localImages = (Wsl "docker images --format '{{.Repository}}:{{.Tag}}' | grep '^aias-local/'").Trim()
if ($localImages) {
  Add-Check $true "Local Space builds present" (($localImages -split "`n") -join "; ")
  $running = (Wsl "docker ps --filter label=aias.kind=space --format '{{.Names}} {{.Status}}'").Trim()
  Add-Check ([bool]$running) "A locally built Space is running" $(if ($running) { $running } else { "none running" })
} else {
  Add-Check $true "Local Space builds" "none yet (Browse -> Build locally on a gradio Space, then rerun)"
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
$host1 = $env:COMPUTERNAME
$os = (Get-CimInstance Win32_OperatingSystem).Caption
"## Vendor verification: $host1 ($os), vendor $effective"
""
foreach ($r in $results) {
  $box = if ($r.ok) { "[x]" } else { "[ ]" }
  $detail = if ($r.detail) { ": $($r.detail)" } else { "" }
  "- $box $($r.label)$detail"
}
""
$failed = @($results | Where-Object { -not $_.ok }).Count
"$($results.Count) checks, $failed failed"
exit $(if ($failed -gt 0) { 1 } else { 0 })
