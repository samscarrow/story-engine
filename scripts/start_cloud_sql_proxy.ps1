param(
  [string]$ConnectionName = "iconic-medley-470519-p1:us-central1:postgres-free-tier",
  [int]$Port = 5433,
  [switch]$Iam
)

if (-not (Get-Command cloud-sql-proxy -ErrorAction SilentlyContinue) -and -not (Test-Path "./cloud-sql-proxy.exe")) {
  Write-Host "cloud-sql-proxy not found in PATH or current directory." -ForegroundColor Yellow
  Write-Host "Download from: https://cloud.google.com/sql/docs/mysql/sql-proxy" -ForegroundColor Yellow
  exit 1
}

$proxy = (Get-Command cloud-sql-proxy -ErrorAction SilentlyContinue)?.Source
if (-not $proxy) { $proxy = "./cloud-sql-proxy.exe" }

$args = @("--port", "$Port")
if ($Iam) { $args += "--auto-iam-authn" }
$args += $ConnectionName

Write-Host "Starting Cloud SQL Auth Proxy on port $Port for $ConnectionName ..." -ForegroundColor Cyan
& $proxy @args
