import asyncio
from datetime import datetime, timedelta
import logging

from app.database.connection import DbConfig, connect_with_retry
from app.core.ml.ai.ai_service import AIService

logger = logging.getLogger(__name__)


async def run_scheduled_tasks(db_config: DbConfig):
    """Main function to run all scheduled background tasks."""
    logger.info("Starting scheduled background tasks...")

    ai_service = AIService(db_config)

    while True:
        try:
            # Run every minute
            # Streaks are updated on user activity, not a global sweep

            # Run every 5 minutes
            if datetime.now().minute % 5 == 0:
                await process_market_sentiment(ai_service)

            # Run every hour - these tasks are now server-side
            # if datetime.now().minute == 0:
            #     await update_all_leaderboards(gamification_service)
            #     await calculate_all_user_metrics(analytics_service)

            # Run daily at a specific time (e.g., 00:00 UTC)
            if datetime.now().hour == 0 and datetime.now().minute == 0:
                await reset_daily_metrics(db_config)
                
                # Run archival job
                from app.services.archival_service import get_archival_service
                archival_service = get_archival_service()
                await archival_service.run_archival_job(older_than_days=7)

        except Exception as e:
            logger.error(f"Error in scheduled task: {e}", exc_info=True)

        await asyncio.sleep(60)  # Wait for 1 minute


# The following functions are now server-side responsibilities or deprecated from client
# async def update_user_streaks(gamification_service: GamificationService):
#     """Update streaks for all active users."""
#     pass


async def process_market_sentiment(ai_service: AIService):
    """Process market sentiment for key assets."""
    logger.info("Processing market sentiment... Currently disabled for future implementation.")
    # This is a placeholder. Actual implementation would involve:
    # 1. Fetching relevant assets (e.g., top 10 crypto, top 10 stocks)
    # 2. Fetching recent news/social media data for these assets
    # 3. Calling ai_service.analyze_sentiment_cached for each piece of content
    # 4. Aggregating sentiment and storing in market_sentiment_daily

    # Example: Analyze sentiment for a dummy text
    # await ai_service.analyze_sentiment_cached("Bitcoin is soaring today!", user_id=None)
    pass


# async def update_all_leaderboards(gamification_service: GamificationService):
#     """Update all leaderboards."""
#     pass

# async def calculate_all_user_metrics(analytics_service: AnalyticsService):
#     """Calculate analytics metrics for all users."""
#     pass


async def reset_daily_metrics(db_config: DbConfig):
    """Reset daily metrics and perform daily maintenance."""
    logger.info("Resetting daily metrics...")
    conn = None
    try:
        conn = await connect_with_retry(db_config, db_config.database_name)
        # Reset current_monthly_cost for users if it's the start of a new month
        if datetime.now().day == 1:
            await conn.execute(
                """
                UPDATE user_ai_preferences
                SET current_monthly_cost = 0
                WHERE current_monthly_cost > 0;
            """
            )
        logger.info("Daily metrics reset.")
    except Exception as e:
        logger.error(f"Error resetting daily metrics: {e}", exc_info=True)
    finally:
        if conn:
            await conn.close()
