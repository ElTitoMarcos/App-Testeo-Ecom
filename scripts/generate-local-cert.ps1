$certDir = Join-Path $PSScriptRoot "..\certs"
New-Item -ItemType Directory -Force -Path $certDir | Out-Null
if (Get-Command mkcert -ErrorAction SilentlyContinue) {
  mkcert -cert-file (Join-Path $certDir "dev-cert.pem") -key-file (Join-Path $certDir "dev-key.pem") 127.0.0.1 localhost
} else {
  openssl req -x509 -newkey rsa:2048 -nodes -keyout (Join-Path $certDir "dev-key.pem") -out (Join-Path $certDir "dev-cert.pem") -days 365 -subj "/CN=localhost"
}
Write-Output "Certificates generated in $certDir"
