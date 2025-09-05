def analyze(product: dict) -> dict:
    """Analyze product heuristically based on basic fields."""
    score = product.get('score', 0)
    trending = max(0, min(5, round((product.get('trend_signal', 0) or score/20))))
    return {
        "producto": {
            "nombre": product.get("name", ""),
            "categoria": product.get("category", ""),
            "precio_mercado": {"min": 0, "max": 0},
            "proveedores": []
        },
        "demanda_y_tendencia": {
            "senales": [],
            "estacionalidad": "",
            "sparkline": [],
            "busqueda_intencion": []
        },
        "competencia": {
            "principales": [],
            "saturacion": "media",
            "puntos_diferenciacion": []
        },
        "social_proof": {
            "yt": {"videos": 0, "avg_views": 0, "eng_rate": 0},
            "reddit": {"hilos": 0, "sentimiento": ""},
            "amazon": {
                "avg_rating": product.get("rating", 0),
                "n_reviews": product.get("reviews", 0)
            },
            "otros": []
        },
        "logistica_y_riesgos": {
            "peso_volumen": "",
            "fragilidad": "",
            "variantes": "",
            "riesgos_legales": [],
            "politicas_plataformas": []
        },
        "unit_economics": {
            "precio_objetivo": product.get("price", 0),
            "costo_estimado": 0,
            "margen_bruto": 0,
            "roi_ads_estimado": 0
        },
        "insights_creativos": {
            "angulos": [],
            "promesas_evitar": [],
            "objeciones_y_respuestas": []
        },
        "pros": [],
        "contras": [],
        "veredicto": {
            "apto_para_test": True,
            "prioridad": "media",
            "razonamiento": ""
        },
        "trendingScore": trending,
        "fuentes": [
            {"tipo": "youtube", "omitida": True},
            {"tipo": "reddit", "omitida": True},
            {"tipo": "amazon", "omitida": True}
        ]
    }
