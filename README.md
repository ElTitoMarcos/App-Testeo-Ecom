# App-Testeo-Ecom

## Arranque silencioso en Windows

Doble clic en `run_silent.vbs` para iniciar la aplicación sin ventana de consola. Tras un segundo, se abrirá automáticamente el navegador en `http://127.0.0.1:8000`.

Alternativa de empaquetado:

```bash
pyinstaller --noconfirm --onefile --windowed product_research_app/web_app.py
```
