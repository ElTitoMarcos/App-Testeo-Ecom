# Ecom Testing App

Aplicación de investigación de productos con servidor HTTP sencillo.

## Desarrollo

```bash
python -m product_research_app.web_app
```

El servidor expone una ruta de salud en `http://127.0.0.1:8000/health`.

Para verificar el entorno dev puede ejecutarse:
```bash
scripts/dev-check.sh
```

Para habilitar HTTPS local consulta [docs/dev-https.md](docs/dev-https.md).
