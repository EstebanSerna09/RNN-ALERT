# ============================================================
# LIMPIAR PROYECTO RNN-ALERT
# Limpieza segura para entregar/comprimir el proyecto.
# IMPORTANTE: este script NO borra outputs ni modelos entrenados.
# Conserva:
#   - outputs\modelos
#   - outputs\modelos_ml
#   - outputs\intermedios
#   - outputs\resultados
#   - outputs\auditoria
# ============================================================
# Uso:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\limpiar_proyecto.ps1
# ============================================================

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " LIMPIEZA SEGURA DEL PROYECTO RNN-ALERT" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$Proyecto = Get-Location
Write-Host "`nRuta actual del proyecto:" -ForegroundColor Yellow
Write-Host $Proyecto

$Confirmacion = Read-Host "`nEste proceso eliminará venv, cachés, logs y temporales, pero conservará outputs y modelos entrenados. ¿Deseas continuar? (S/N)"
if ($Confirmacion -ne "S" -and $Confirmacion -ne "s") {
    Write-Host "Proceso cancelado." -ForegroundColor Red
    exit
}

Write-Host "`nCalculando peso inicial..." -ForegroundColor Yellow
$PesoInicial = (Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
$PesoInicialMB = [math]::Round($PesoInicial / 1MB, 2)
Write-Host "Peso inicial: $PesoInicialMB MB" -ForegroundColor Green

# Entornos virtuales
Write-Host "`nEliminando entornos virtuales..." -ForegroundColor Yellow
$Entornos = @("venv", ".venv", "env", "ENV")
foreach ($entorno in $Entornos) {
    if (Test-Path $entorno) {
        Write-Host "Eliminando $entorno"
        Remove-Item -Recurse -Force $entorno -ErrorAction SilentlyContinue
    }
}

# Carpetas temporales que NO son outputs del proyecto
Write-Host "`nEliminando carpetas temporales..." -ForegroundColor Yellow
$CarpetasTemporales = @(
    "runs",
    "logs",
    "checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".streamlit\cache"
)
foreach ($carpeta in $CarpetasTemporales) {
    if (Test-Path $carpeta) {
        Write-Host "Eliminando $carpeta"
        Remove-Item -Recurse -Force $carpeta -ErrorAction SilentlyContinue
    }
}

# Caches Python/Jupyter
Write-Host "`nEliminando cachés de Python y Jupyter..." -ForegroundColor Yellow
Get-ChildItem -Recurse -Directory -Force -Include "__pycache__", ".ipynb_checkpoints" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Archivos temporales generales, sin tocar modelos ni outputs
Write-Host "`nEliminando archivos temporales generales..." -ForegroundColor Yellow
$PatronesEliminar = @("*.pyc", "*.pyo", "*.log", "*.tmp", "*.temp", ".DS_Store", "Thumbs.db")
foreach ($patron in $PatronesEliminar) {
    Get-ChildItem -Recurse -File -Include $patron -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\outputs\\" } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

# Asegurar estructura base sin borrar contenido existente
Write-Host "`nVerificando estructura base..." -ForegroundColor Yellow
$CarpetasBase = @(
    "outputs",
    "outputs\modelos",
    "outputs\modelos_ml",
    "outputs\intermedios",
    "outputs\resultados",
    "outputs\auditoria",
    "uploads",
    "assets",
    "docs"
)
foreach ($carpeta in $CarpetasBase) {
    New-Item -ItemType Directory -Force $carpeta | Out-Null
}

Write-Host "`nCalculando peso final..." -ForegroundColor Yellow
$PesoFinal = (Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
$PesoFinalMB = [math]::Round($PesoFinal / 1MB, 2)
Write-Host "Peso final: $PesoFinalMB MB" -ForegroundColor Green

Write-Host "`nLimpieza segura completada." -ForegroundColor Cyan
Write-Host "Modelos entrenados y outputs fueron conservados." -ForegroundColor Green
