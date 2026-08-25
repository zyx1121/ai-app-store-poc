# Probe GPUs and NPUs on the Windows host. Printed as one JSON line for the Rust side.
$ErrorActionPreference = "SilentlyContinue"
$classKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
$regEntries = Get-ChildItem $classKey | Where-Object { $_.PSChildName -match '^\d{4}$' } | ForEach-Object { Get-ItemProperty $_.PSPath }
$gpus = Get-CimInstance Win32_VideoController | ForEach-Object {
  $name = $_.Name
  $vram = $null
  foreach ($p in $regEntries) {
    if ($p.DriverDesc -eq $name -and $p.'HardwareInformation.qwMemorySize') { $vram = [math]::Round($p.'HardwareInformation.qwMemorySize' / 1MB) }
  }
  if (-not $vram -and $_.AdapterRAM) { $vram = [math]::Round($_.AdapterRAM / 1MB) }
  [pscustomobject]@{ name = $name; vendor = $_.AdapterCompatibility; driver = $_.DriverVersion; pnp = $_.PNPDeviceID; vram_mb = $vram }
}
$npus = Get-PnpDevice -PresentOnly | Where-Object {
  $_.FriendlyName -match 'AI Boost|IPU Device|Hexagon|Neural Process|\bNPU\b|Ryzen AI'
} | ForEach-Object { [pscustomobject]@{ name = $_.FriendlyName; class = $_.Class; status = $_.Status } }
[pscustomobject]@{ gpus = @($gpus); npus = @($npus) } | ConvertTo-Json -Depth 4 -Compress
