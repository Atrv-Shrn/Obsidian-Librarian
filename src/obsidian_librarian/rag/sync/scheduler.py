"""APScheduler-driven periodic vault re-sync.

Boots alongside the agent API inside the container (a supervisord program). Every
``SYNC_INTERVAL_MINUTES`` it calls :func:`rag.ingest.sync_vault` to pick up new/changed/deleted
notes. If ``SYNC_ON_START`` is set, it runs one sync immediately at boot so the index is fresh
before the first query. Failures are logged, never fatal — the next tick retries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from ...config import get_settings

log = logging.getLogger(__name__)


def _job() -> None:
    from ..ingest import sync_vault

    try:
        report = sync_vault()
        log.info(
            "scheduled sync: +a=%d ~m=%d -d=%d dup=%d errs=%d",
            report.added, report.modified, report.deleted, report.skipped_dup, len(report.errors),
        )
        for e in report.errors:
            log.warning("sync error: %s", e)
    except Exception as e:  # never crash the scheduler
        log.exception("scheduled sync failed: %s", e)


def build_scheduler() -> BlockingScheduler:
    s = get_settings()
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(
        _job,
        IntervalTrigger(minutes=s.sync_interval_minutes),
        id="vault-sync",
        # Defer the first interval tick to one period from now. ``main()`` already runs an
        # on-start sync when SYNC_ON_START is set, so letting the interval job fire
        # immediately on start would double-run it; an explicit next_run_time avoids that
        # and removes the previous tautological ternary (both branches were None).
        next_run_time=datetime.now(timezone.utc) + timedelta(minutes=s.sync_interval_minutes),
    )
    return sched


def main() -> None:  # pragma: no cover - process entry
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    s = get_settings()
    if s.sync_on_start:
        log.info("running on-start sync…")
        _job()
    sched = build_scheduler()
    log.info("starting scheduler (every %d min)", s.sync_interval_minutes)
    sched.start()


if __name__ == "__main__":  # pragma: no cover
    main()