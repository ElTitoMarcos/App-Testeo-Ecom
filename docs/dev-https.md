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
3. Confía el certificado generado (`certs/dev-cert.pem`) en tu navegador si es necesario.

El modo HTTP sigue disponible simplemente ejecutando:
```bash
python -m product_research_app.web_app
```
