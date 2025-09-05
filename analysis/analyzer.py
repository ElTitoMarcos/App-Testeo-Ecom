import os

from product_research_app import social_scraper


def analyze(product: dict) -> dict:
    """Analyze product using lightweight web signals and heuristics."""
    score = product.get("score", 0)
    trending = max(0, min(5, round((product.get("trend_signal", 0) or score / 20))))

    name = (
        product.get("name")
        or product.get("title")
        or product.get("Product Name")
        or ""
    )
    reddit_data = social_scraper.fetch_reddit_posts(name) if name else {"comments": [], "post_count": 0}
    yt_key = os.environ.get("YOUTUBE_API_KEY")
    youtube_data = social_scraper.fetch_youtube_comments(yt_key, name) if name else {"comments": [], "video_count": 0}
    web_data = social_scraper.fetch_web_reviews(name) if name else {"comments": [], "sources": []}
    comments = reddit_data["comments"] + youtube_data["comments"] + web_data["comments"]
    keywords = social_scraper.extract_keywords(comments)
    summary = social_scraper.summarize_comments(comments)
    pros = summary.get("pros", [])
    contras = summary.get("contras", [])
    related = summary.get("productos_relacionados", [])
    repeated = social_scraper.find_repeated_comments(comments)

    return {
        "producto": {
            "nombre": product.get("name", ""),
            "categoria": product.get("category", ""),
            "precio_mercado": {"min": 0, "max": 0},
            "proveedores": [],
        },
        "demanda_y_tendencia": {
            "senales": [],
            "estacionalidad": "",
            "sparkline": [],
            "busqueda_intencion": [],
        },
        "competencia": {
            "principales": [],
            "saturacion": "media",
            "puntos_diferenciacion": [],
        },
        "social_proof": {
            "yt": {
                "videos": youtube_data.get("video_count", 0),
                "avg_views": 0,
                "eng_rate": 0,
            },
            "reddit": {
                "hilos": reddit_data.get("post_count", 0),
                "sentimiento": "",
            },
            "amazon": {
                "avg_rating": product.get("rating", 0),
                "n_reviews": product.get("reviews", 0),
            },
            "otros": web_data.get("sources", []),
        },
        "logistica_y_riesgos": {
            "peso_volumen": "",
            "fragilidad": "",
            "variantes": "",
            "riesgos_legales": [],
            "politicas_plataformas": [],
        },
        "unit_economics": {
            "precio_objetivo": product.get("price", 0),
            "costo_estimado": 0,
            "margen_bruto": 0,
            "roi_ads_estimado": 0,
        },
        "insights_creativos": {
            "angulos": [],
            "promesas_evitar": [],
            "objeciones_y_respuestas": [],
        },
        "pros": pros,
        "contras": contras,
        "palabras_clave": keywords,
        "productos_relacionados": related,
        "comentarios_frecuentes": repeated,
        "veredicto": {
            "apto_para_test": True,
            "prioridad": "media",
            "razonamiento": "",
        },
        "trendingScore": trending,
        "fuentes": [
            {"tipo": "youtube", "omitida": youtube_data.get("video_count", 0) == 0},
            {"tipo": "reddit", "omitida": reddit_data.get("post_count", 0) == 0},
            {"tipo": "web", "omitida": len(web_data.get("sources", [])) == 0},
            {"tipo": "amazon", "omitida": True},
        ],
    }
