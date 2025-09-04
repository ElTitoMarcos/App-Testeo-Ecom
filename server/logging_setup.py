import logging, os
from logging.handlers import RotatingFileHandler

def setup_logging(log_path='./logs/app.log'):
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
  return os.path.abspath(log_path)
