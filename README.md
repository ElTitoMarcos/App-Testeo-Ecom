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

## Mantenimiento SQLite — Reinicio total (solo desarrollo)

El proyecto incluye un comando para vaciar por completo la tabla de productos y
reiniciar el contador de IDs de la base de datos SQLite. Está pensado solo para
entornos de desarrollo.

```
python -m product_research_app.maintenance purge-and-reset \
  --yes --i-know-what-im-doing --no-prompt
```

- Crea un respaldo automático en `backups/` antes de realizar cambios.
- Requiere que `APP_ENV` o `ENV` sea `development` o `local`.
- En otros entornos se rechazará salvo que se añada `--dangerously-on-prod`
  junto con las confirmaciones dobles.
- Bandera `--force` ignora claves foráneas (imprime advertencias visibles).

### Uso en CI

En pipelines de integración continua puede ejecutarse en modo no interactivo
añadiendo la bandera `--no-prompt` y asegurando que las variables de entorno
correspondan al entorno de desarrollo.

### Recuperación ante fallos

Si algo sale mal, restaura la copia de seguridad reemplazando el archivo de la
base de datos por el último backup generado en `backups/`.
