# Title Analyzer

This project exposes a `/api/analyze/titles` endpoint and a small web UI panel to inspect product titles.

## Usage

### Run the app

```bash
python -m product_research_app.web_app
```
Open [http://localhost:8000](http://localhost:8000) in a browser and use the **Analizador de Títulos** panel to upload a CSV or JSON file.

### API example

POST a JSON array of products:

```bash
curl -X POST http://localhost:8000/api/analyze/titles \
  -H 'Content-Type: application/json' \
  -d '[{"title": "Waterproof Magnetic Case for iPhone 15, 2 Pack, 64oz", "price": 19.99}]'
```

The response contains a `signals` block, risk `flags`, a Spanish `summary` and a `titleScore`.

### Python module

You can also call the analyzer directly:

```python
from product_research_app.title_analyzer import analyze_titles

items = [{"title": "Premium Best Kitchen Set", "price": 25}]
print(analyze_titles(items))
```

### Tests

Run the minimal unit tests with:

```bash
python3 -m pytest
```
