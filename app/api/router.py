from fastapi import APIRouter

from app.api.routes import analytics, auth, datasets, users

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(datasets.router)
api_router.include_router(analytics.router)
