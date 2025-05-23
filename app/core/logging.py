import sys
from loguru import logger
import os

# 로그 레벨 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 로거 설정
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# 노트북 디버깅 함수를 로거로 변환
def log_success(message):
    logger.success(f"✅ {message}")

def log_info(message):
    logger.info(f"ℹ️ {message}")

def log_warning(message):
    logger.warning(f"⚠️ {message}")

def log_error(message):
    logger.error(f"❌ {message}")