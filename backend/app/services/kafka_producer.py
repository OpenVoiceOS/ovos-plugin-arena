"""
Kafka producer service for OVOS Plugin Arena.
Singleton pattern with connection pooling and automatic retry logic.
"""

import asyncio
import json
import logging
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError

from app.core.config import settings
from app.schemas.kafka_messages import (
    BattleExecutionRequested,
    serialize_message,
)

logger = logging.getLogger(__name__)


class KafkaProducerService:
    """
    Singleton Kafka producer with connection pooling.
    Handles automatic reconnection and message publishing.
    """

    _instance: "KafkaProducerService | None" = None
    _producer: AIOKafkaProducer | None = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "KafkaProducerService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def start(self) -> None:
        """Initialize and start the Kafka producer."""
        if self._producer is not None:
            logger.warning("Kafka producer already started")
            return

        async with self._lock:
            if self._producer is not None:
                return

            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    compression_type="gzip",
                    # Idempotence configuration
                    enable_idempotence=True,
                    acks="all",
                    max_in_flight_requests_per_connection=1,
                    # Retry configuration
                    retries=5,
                    retry_backoff_ms=100,
                    # Timeout configuration
                    request_timeout_ms=30000,
                    # Batch configuration for better throughput
                    linger_ms=10,
                    batch_size=16384,
                )
                await self._producer.start()
                logger.info(
                    "Kafka producer started successfully: %s",
                    settings.KAFKA_BOOTSTRAP_SERVERS,
                )
            except KafkaConnectionError as e:
                logger.error("Failed to connect to Kafka: %s", e)
                self._producer = None
                raise
            except Exception as e:
                logger.error("Failed to start Kafka producer: %s", e)
                self._producer = None
                raise

    async def stop(self) -> None:
        """Stop and cleanup the Kafka producer."""
        if self._producer is None:
            return

        async with self._lock:
            if self._producer is None:
                return

            try:
                await self._producer.stop()
                logger.info("Kafka producer stopped successfully")
            except Exception as e:
                logger.error("Error stopping Kafka producer: %s", e)
            finally:
                self._producer = None

    async def _ensure_started(self) -> None:
        """Ensure producer is started, start if needed."""
        if self._producer is None:
            await self.start()

    async def publish_message(
        self, topic: str, message: dict[str, Any], key: str | None = None
    ) -> None:
        """
        Publish a message to Kafka topic.

        Args:
            topic: Kafka topic name
            message: Message payload as dict
            key: Optional message key for partitioning

        Raises:
            KafkaError: If message publishing fails after retries
        """
        await self._ensure_started()

        if self._producer is None:
            raise RuntimeError("Kafka producer not available")

        try:
            key_bytes = key.encode("utf-8") if key else None
            await self._producer.send_and_wait(
                topic=topic, value=message, key=key_bytes
            )
            logger.info(
                "Published message to topic=%s, key=%s",
                topic,
                key if key else "None",
            )
        except KafkaError as e:
            logger.error("Failed to publish message to topic=%s: %s", topic, e)
            raise
        except Exception as e:
            logger.error("Unexpected error publishing to topic=%s: %s", topic, e)
            raise

    async def publish_battle_execution_requested(
        self, message: BattleExecutionRequested
    ) -> None:
        """
        Publish battle execution request to Kafka.

        Args:
            message: BattleExecutionRequested message

        This is the primary entry point for requesting battle execution.
        Workers consume from this topic to execute plugin battles.
        """
        message_dict = serialize_message(message)
        await self.publish_message(
            topic=settings.KAFKA_TOPIC_BATTLE_EXECUTION,
            message=message_dict,
            key=str(message.battle_id),  # Ensure same battle goes to same partition
        )

    async def publish_vote_submitted(
        self, vote_id: str, battle_id: str, user_id: str, result: str,
        competitor_a_id: str, competitor_b_id: str
    ) -> None:
        """
        Publish vote submission event to Kafka.

        Args:
            vote_id: UUID of the vote
            battle_id: UUID of the battle
            user_id: UUID of the user who voted
            result: Vote result (candidate_1, candidate_2, tie, both_wrong)
            competitor_a_id: UUID of competitor A
            competitor_b_id: UUID of competitor B

        Triggers ELO calculation by workers.
        """
        message = {
            "vote_id": vote_id,
            "battle_id": battle_id,
            "user_id": user_id,
            "result": result,
            "competitor_a_id": competitor_a_id,
            "competitor_b_id": competitor_b_id,
            "submitted_at": None,  # Set by Kafka timestamp
        }
        await self.publish_message(
            topic=settings.KAFKA_TOPIC_VOTE_SUBMITTED,
            message=message,
            key=battle_id,  # Partition by battle_id for ordering
        )


# Global singleton instance
kafka_producer = KafkaProducerService()


async def get_kafka_producer() -> KafkaProducerService:
    """
    Dependency for FastAPI routes.
    Ensures producer is started before use.
    """
    await kafka_producer._ensure_started()
    return kafka_producer
