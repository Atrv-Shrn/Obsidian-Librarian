"""Sync subpackage — keeps the index aligned with the vault.

* :mod:`watermarks` — SQLite table ``path → sha256, mtime``; the single source of
  truth for what is indexed. Sync diffs the filesystem against this table.
* :mod:`scheduler`  — APScheduler job that periodically runs a full sync.
"""