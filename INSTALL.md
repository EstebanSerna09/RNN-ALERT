# Instalación y Ejecución — RNN-ALERT

## WINDOWS (PowerShell)

```powershell
# 1. Crear entorno virtual
python -m venv venv

# 2. Activar entorno virtual
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

## macOS / Linux (Terminal)

```bash
# 1. Crear entorno virtual
python3 -m venv venv

# 2. Activar entorno virtual
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

## Solución de errores comunes

### PowerShell: "scripts is disabled on this system"
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### pip no reconocido
```powershell
python -m pip install -r requirements.txt
```

### TensorFlow no instala en Python 3.13+
```powershell
# Instalar Python 3.11 (compatible con TensorFlow)
# https://www.python.org/downloads/release/python-31110/
```
