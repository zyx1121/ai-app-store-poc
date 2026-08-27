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
# Virtualization: WSL2 needs the Windows hypervisor, which needs VT-x / AMD-V
# enabled in UEFI firmware. `vt_supported` is the silicon capability,
# `vt_firmware_enabled` the BIOS switch, `hypervisor_present` means it is already
# running (so it is usable regardless of the other two).
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$cs = Get-CimInstance Win32_ComputerSystem
$virt = [pscustomobject]@{
  vt_supported        = [bool]$cpu.VMMonitorModeExtensions
  vt_firmware_enabled = [bool]$cpu.VirtualizationFirmwareEnabled
  hypervisor_present  = [bool]$cs.HypervisorPresent
}
# Integrated GPUs and NPUs address system RAM (WDDM shared GPU memory), so the
# Rust side needs the physical total to size their budget; the registry only
# reports the small dedicated carve-out for them.
$totalRam = if ($cs.TotalPhysicalMemory) { [math]::Round($cs.TotalPhysicalMemory / 1MB) } else { $null }
[pscustomobject]@{ gpus = @($gpus); npus = @($npus); virtualization = $virt; total_ram_mb = $totalRam } | ConvertTo-Json -Depth 4 -Compress
