from product_research_app.services.ai_pipeline import run_ai_pipeline

if __name__ == "__main__":
    import os

    limit_env = os.getenv("PRAPP_PIPELINE_LIMIT")
    limit = int(limit_env) if limit_env else None
    run_ai_pipeline(limit=limit)
