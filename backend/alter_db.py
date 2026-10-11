import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        from sqlalchemy import text
        await conn.execute(text("ALTER TABLE analysis_jobs ALTER COLUMN job_description_url TYPE TEXT;"))
    print("Column altered successfully!")

if __name__ == "__main__":
    asyncio.run(main())
