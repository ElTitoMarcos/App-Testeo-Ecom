import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def enrich_with_ai_batched(products: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """Placeholder IA enrichment using batch-columns endpoint.

    This function performs a single request per batch of up to 25 products.
    The implementation here is a minimal stub that simply logs the
    invocation and returns an empty list, ensuring the rest of the pipeline
    continues to operate. It can be extended later to perform real HTTP
    requests to the ``/api/ia/batch-columns`` endpoint.
    """

    if not products:
        return []
    logger.info("enrich_with_ai_batched called for %d products", len(products))
    # Real implementation would go here
    return []
