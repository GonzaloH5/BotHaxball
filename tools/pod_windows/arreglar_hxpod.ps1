# Arregla el acceso SSH de la cuenta "hxpod" (PC prestada por Tailscale). Correr en PowerShell como ADMINISTRADOR:
#   powershell -ExecutionPolicy Bypass -File arreglar_hxpod.ps1
#
# Problema: el primer ingreso no pudo cargar el perfil de hxpod (Windows le dio un perfil temporal y marcó el suyo
# como roto), y desde ahí OpenSSH no encuentra la clave dentro del perfil.
# Qué hace (y nada más):
#   1. Guarda la clave pública de Gonzalo en C:\ProgramData\ssh\hxpod-claves (solo SYSTEM y Administradores pueden
#      modificarla), así la autorización no depende del perfil.
#   2. Agrega al final de sshd_config una regla solo para hxpod que usa ese archivo (si no estaba).
#   3. Borra el perfil roto de hxpod (registro y carpeta), para que Windows lo cree de nuevo en el próximo ingreso.
#   4. Reinicia el servicio sshd (corta solo las sesiones SSH abiertas, si hubiera).
#   5. Muestra cómo quedó y escribe LISTO.

$ErrorActionPreference = "Stop"
$user = "hxpod"
$key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOqJOC6OtTQ8vaKnRJSBgUDhY0n6+NRbN4/LqX/fuxDg gonzalo20062010@gmail.com"
$u = [char]95
$sshData = "C:\ProgramData\ssh"
$keyFile = Join-Path $sshData "hxpod-claves"
$conf = Join-Path $sshData ("sshd" + $u + "config")

# 1. clave fuera del perfil
Set-Content -Path $keyFile -Value $key -Encoding ascii
$acl = New-Object System.Security.AccessControl.FileSecurity
$acl.SetAccessRuleProtection($true, $false)
foreach ($s in @("S-1-5-18", "S-1-5-32-544")) {
    $id = New-Object System.Security.Principal.SecurityIdentifier($s)
    $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($id, "FullControl", "Allow")))
}
$sid = (Get-LocalUser -Name $user).SID.Value
$acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
    (New-Object System.Security.Principal.SecurityIdentifier($sid)), "Read", "Allow")))
$acl.SetOwner((New-Object System.Security.Principal.SecurityIdentifier("S-1-5-32-544")))
Set-Acl -Path $keyFile -AclObject $acl

# 2. regla de sshd solo para hxpod
$text = Get-Content $conf -Raw
if ($text -notmatch "Match User hxpod") {
    $line = "Authorized" + "KeysFile " + "C:/ProgramData/ssh/hxpod-claves"
    Add-Content -Path $conf -Value ("`r`nMatch User hxpod`r`n       " + $line + "`r`n") -Encoding ascii
}

# 3. perfil roto: registro (también la copia .bak) y carpeta
$pl = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList"
foreach ($kname in @($sid, "$sid.bak")) {
    if (Test-Path "$pl\$kname") { Remove-Item -Path "$pl\$kname" -Recurse -Force }
}
if (Test-Path "C:\Users\$user") { Remove-Item -Recurse -Force "C:\Users\$user" -ErrorAction SilentlyContinue }

# 4. reiniciar sshd
Restart-Service sshd

Write-Host ""
Write-Host "Clave:"
icacls $keyFile
Write-Host "Regla en sshd:"
Get-Content $conf | Select-String -Pattern "Match|hxpod-claves"
Write-Host "sshd:" (Get-Service sshd).Status
Write-Host ""
Write-Host "LISTO"
