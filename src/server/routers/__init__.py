from src.server.routers.emotions import router as emotions_router
from src.server.routers.labels import router as labels_router
from src.server.routers.quality import api_router as quality_api_router
from src.server.routers.quality import page_router as quality_page_router
from src.server.routers.system import router as system_router
from src.server.routers.tts import router as tts_router
from src.server.routers.voices import router as voices_router

all_routers = [
    system_router, voices_router, tts_router, labels_router, emotions_router,
    quality_api_router, quality_page_router,
]
