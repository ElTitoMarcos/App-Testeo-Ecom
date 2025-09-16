# Import Benchmarking

The `bench_import.py` script generates a synthetic CSV dataset and measures the
end-to-end performance of the `/upload` import pipeline.

## Usage

1. Start the Flask backend (for example: `python -m product_research_app.app`).
2. Run the benchmark:

   ```bash
   python scripts/bench_import.py --base-url http://127.0.0.1:5000 --rows 10000
   ```

The script prints the total elapsed time, backend-reported phase timings, and
the resulting throughput in rows per second. Adjust `--rows` or `--seed` to
stress different dataset sizes while keeping deterministic data generation.
