# Prepara la cuenta "hxpod" para trabajos de HaxballRL por SSH (PC prestada por Tailscale).
# Correr en PowerShell como ADMINISTRADOR:
#   powershell -ExecutionPolicy Bypass -File preparar_hxpod.ps1
#
# Qué hace (y nada más):
#   1. Si la cuenta hxpod no existe, la crea como usuario estándar (sin permisos de administrador).
#   2. Le pone la contraseña que se escriba acá (no hace falta compartirla con nadie).
#   3. Crea su perfil de Windows sin abrir una sesión en el escritorio (si había una carpeta suelta
#      C:\Users\hxpod que no era un perfil, la borra antes, para que el perfil quede en esa ruta).
#   4. Autoriza solo la clave pública de Gonzalo en ~/.ssh/authorized_keys, con los permisos que pide OpenSSH.
#   5. Oculta la cuenta en la pantalla de inicio de Windows.
#   6. Muestra cómo quedó todo y escribe LISTO.

$ErrorActionPreference = "Stop"
$user = "hxpod"
$key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOqJOC6OtTQ8vaKnRJSBgUDhY0n6+NRbN4/LqX/fuxDg gonzalo20062010@gmail.com"

$pw = Read-Host -AsSecureString "Contrasena para $user (la que quieras; no hace falta compartirla)"
if (-not (Get-LocalUser -Name $user -ErrorAction SilentlyContinue)) {
    New-LocalUser -Name $user -Password $pw -PasswordNeverExpires -AccountNeverExpires -Description "Trabajos HaxballRL por SSH" | Out-Null
} else {
    Set-LocalUser -Name $user -Password $pw -PasswordNeverExpires $true
}
$sid = (Get-LocalUser -Name $user).SID.Value

# perfil: si no está registrado, borrar la carpeta suelta y crearlo con un proceso oculto de la cuenta
$prof = Get-CimInstance Win32_UserProfile | Where-Object { $_.SID -eq $sid }
if (-not $prof) {
    $loose = "C:\Users\$user"
    if (Test-Path $loose) { Remove-Item -Recurse -Force $loose }
    $cred = New-Object System.Management.Automation.PSCredential($user, $pw)
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c exit" -Credential $cred -LoadUserProfile -WindowStyle Hidden -WorkingDirectory "C:\Windows\System32" -Wait
    $prof = Get-CimInstance Win32_UserProfile | Where-Object { $_.SID -eq $sid }
}
if (-not $prof) { throw "No se pudo crear el perfil de $user (servicio Inicio de sesion secundario apagado?)" }
$profPath = $prof.LocalPath

# clave autorizada, con permisos estrictos: solo la cuenta, SYSTEM y Administradores
$sshDir = Join-Path $profPath ".ssh"
New-Item -ItemType Directory -Force $sshDir | Out-Null
$keyFile = Join-Path $sshDir ("authorized" + [char]95 + "keys")
Set-Content -Path $keyFile -Value $key -Encoding ascii
$acl = New-Object System.Security.AccessControl.FileSecurity
$acl.SetAccessRuleProtection($true, $false)
foreach ($s in @($sid, "S-1-5-18", "S-1-5-32-544")) {
    $id = New-Object System.Security.Principal.SecurityIdentifier($s)
    $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($id, "FullControl", "Allow")))
}
Set-Acl -Path $keyFile -AclObject $acl

# ocultar la cuenta en la pantalla de inicio
$k = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon\SpecialAccounts\UserList"
if (-not (Test-Path $k)) { New-Item -Path $k -Force | Out-Null }
Set-ItemProperty -Path $k -Name $user -Value 0 -Type DWord

Write-Host ""
Write-Host "Perfil: $profPath"
Write-Host "Clave:  $keyFile"
icacls $keyFile
Write-Host "Administradores:"
Get-LocalGroupMember -SID S-1-5-32-544 | Select-Object -ExpandProperty Name
Write-Host ""
Write-Host "LISTO"
