# Desarrollo con HTTPS opcional

El servidor se ejecuta en HTTP por defecto. Si deseas probar HTTPS de manera local:

1. Genera un certificado auto-firmado:
   ```bash
   scripts/generate-local-cert.sh
   ```
   En Windows PowerShell:
   ```powershell
   scripts\generate-local-cert.ps1
   ```
2. Inicia el servidor habilitando HTTPS:
   ```bash
   DEV_HTTPS=true python -m product_research_app.web_app
   ```
   El servidor escuchará en HTTPS en el puerto `8001` (HTTP permanece en `8000`).
3. Confía el certificado generado (`certs/dev-cert.pem`) en tu navegador si es necesario y accede a
   `https://127.0.0.1:8001`.

El modo HTTP sigue disponible simplemente ejecutando:
```bash
python -m product_research_app.web_app
```
