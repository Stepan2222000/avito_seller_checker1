# Database Management Scripts

This directory contains scripts for managing the PostgreSQL database for the Avito Seller Checker project.

## Overview

The project supports two modes of operation:
- **File mode** (default): Tasks and results are managed via text files (`data/urls.txt`, `data/results.json`)
- **Database mode**: Tasks and results are managed via PostgreSQL database for distributed processing

## Enabling Database Mode

To enable database mode, edit `seller_validator/config.py` and set:

```python
use_database: bool = True
```

Database connection settings are already configured in `config.py`:
- Host: `81.30.105.134`
- Port: `5414`
- Database: `avito_seller_checker`
- User: `admin`
- Password: `Password123`

## Available Scripts

### 1. migrate_results.py

Migrates existing results from `data/results.json` to the database. This is a **one-time migration** script.

**Usage:**
```bash
# Migrate from default location (data/results.json)
python scripts/migrate_results.py

# Migrate from custom file
python scripts/migrate_results.py /path/to/results.json
```

**What it does:**
- Reads all results from `results.json`
- Creates a task for each result with `completed` state
- Inserts the result into the `results` table
- Skips duplicates (URLs already in database)

**Example output:**
```
============================================================
Results Migration Tool for Avito Seller Checker
============================================================
Source file: /Users/.../data/results.json

Found 150 results to migrate

Proceed with migration? (yes/no): yes

Connecting to database at 81.30.105.134:5414/avito_seller_checker...

Migrating results...

  Progress: 50/150 (50 migrated, 0 skipped, 0 errors)
  Progress: 100/150 (100 migrated, 0 skipped, 0 errors)

============================================================
Migration completed!
  Total results: 150
  Successfully migrated: 150
  Skipped (duplicates): 0
  Errors: 0
============================================================
```

---

### 2. load_urls.py

Loads seller URLs from a text file into the database tasks table.

**Usage:**
```bash
# Load from default location (data/urls.txt)
python scripts/load_urls.py

# Load from custom file
python scripts/load_urls.py /path/to/urls.txt
```

**Modes:**
- **Add mode** (`a`): Adds URLs to existing tasks, skips duplicates
- **Replace mode** (`r`): **Deletes all pending tasks** and loads new URLs
  - ⚠️ Only deletes `pending` tasks
  - Does NOT touch `in_progress`, `completed`, or `abandoned` tasks
  - Safe for use while workers are running

**Example usage:**
```
============================================================
URL Loader for Avito Seller Checker
============================================================
Source file: /Users/.../data/urls.txt

Read 500 URLs from /Users/.../data/urls.txt

Choose operation mode:
  [a] Add - Add URLs to existing tasks (skip duplicates)
  [r] Replace - Clear pending tasks and load new URLs

Enter mode (a/r): r

WARNING: Replace mode will delete all PENDING tasks!
         (In-progress, completed, and failed tasks will not be touched)
Are you sure you want to continue? (yes/no): yes

Connecting to database at 81.30.105.134:5414/avito_seller_checker...

Clearing pending tasks...
Deleted 120 pending tasks

Replacing 500 URLs...

============================================================
Operation completed successfully!
  Mode: Replace
  Total URLs processed: 500
  New tasks inserted: 480
  Duplicates skipped: 20
============================================================

Current task statistics:
  abandoned: 5
  completed: 150
  in_progress: 8
  pending: 480
```

---

### 3. load_proxies.py

Loads proxy addresses from a text file into the database.

**Usage:**
```bash
# Load from default location (data/proxies.txt)
python scripts/load_proxies.py

# Load from custom file
python scripts/load_proxies.py /path/to/proxies.txt
```

**Modes:**
- **Add mode** (`a`): Adds proxies to existing list, skips duplicates
- **Replace mode** (`r`): **Deletes ALL proxies** (including blocked) and loads new ones
  - ⚠️ Use with caution - clears blocked proxy history

**Example usage:**
```
============================================================
Proxy Loader for Avito Seller Checker
============================================================
Source file: /Users/.../data/proxies.txt

Read 50 proxies from /Users/.../data/proxies.txt

Choose operation mode:
  [a] Add - Add proxies to existing list (skip duplicates)
  [r] Replace - Delete all proxies and load new ones

Enter mode (a/r): a

Connecting to database at 81.30.105.134:5414/avito_seller_checker...

Current proxy statistics:
  available: 35
  blocked: 15
  total: 50

Adding 50 proxies...

============================================================
Operation completed successfully!
  Mode: Add
  Total proxies processed: 50
  New proxies inserted: 5
  Duplicates skipped: 45
============================================================

Current proxy statistics:
  available: 40
  blocked: 15
  total: 55
```

---

### 4. recover_stuck_tasks.py

Recovers tasks that are stuck in `in_progress` state for too long (default: 60 minutes). This can happen when a worker crashes or is terminated unexpectedly.

**Usage:**
```bash
# Dry run (preview only, no changes)
python scripts/recover_stuck_tasks.py

# With custom timeout (90 minutes)
python scripts/recover_stuck_tasks.py --timeout 90

# Actually recover tasks (default timeout: 60 minutes)
python scripts/recover_stuck_tasks.py --apply

# Custom timeout + apply changes
python scripts/recover_stuck_tasks.py --timeout 30 --apply
```

**Modes:**
- **Dry run** (default): Shows which tasks would be recovered without making changes
- **Apply mode** (`--apply`): Actually recovers the tasks by setting their state back to `pending`

**Example usage:**
```
============================================================
Stuck Tasks Recovery Tool
============================================================
Database: 81.30.105.134:5414/avito_seller_checker
Timeout: 60 minutes
Mode: DRY RUN (no changes)

Found 5 stuck tasks:

URL                                                          Stuck For            Attempt
------------------------------------------------------------------------------------------
https://www.avito.ru/user/abc123/profile                     1:23:45              2
https://www.avito.ru/user/def456/profile                     2:10:30              1
https://www.avito.ru/user/ghi789/profile                     1:05:12              3
...

⚠️  DRY RUN MODE - No changes will be made

To actually recover these tasks, run:
  python scripts/recover_stuck_tasks.py --timeout 60 --apply
```

**When to use:**
- After a worker crash or unexpected termination
- When tasks are stuck for a long time without progress
- As a regular maintenance task (e.g., daily cron job)

**Recommended timeout values:**
- **30-60 minutes** for normal operations
- **5-10 minutes** if you know workers crashed recently
- **2-3 hours** if workers are processing very complex tasks

---

## Database Schema

The database has the following tables:

### `tasks`
Stores URLs to be processed.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `url` | TEXT | Seller URL (unique) |
| `payload` | JSONB | Metadata (source, timestamps) |
| `state` | TEXT | pending, in_progress, completed, failed, abandoned |
| `attempt` | INTEGER | Current attempt number |
| `max_attempts` | INTEGER | Maximum attempts (default: 4) |
| `last_proxy` | TEXT | Last used proxy |
| `enqueued_at` | TIMESTAMP | When task was added |
| `updated_at` | TIMESTAMP | Last update time |
| `started_at` | TIMESTAMP | When processing started |
| `completed_at` | TIMESTAMP | When processing finished |

### `results`
Stores validation results.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `task_id` | UUID | Foreign key to tasks.id |
| `seller_url` | TEXT | Seller URL |
| `verdict` | TEXT | passed, failed, error |
| `reason` | TEXT | Explanation |
| `confidence` | REAL | Confidence score (0.0-1.0) |
| `flags` | TEXT[] | Additional flags |
| `items` | JSONB | Item-level decisions |
| `llm_attempts` | INTEGER | Number of LLM attempts |
| `created_at` | TIMESTAMP | When result was created |

### `proxies`
Stores proxy addresses and their status.

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `address` | TEXT | Proxy address (unique) |
| `is_blocked` | BOOLEAN | Whether proxy is blocked |
| `blocked_reason` | TEXT | Reason for blocking |
| `blocked_at` | TIMESTAMP | When proxy was blocked |
| `in_use` | BOOLEAN | Whether proxy is currently in use |
| `last_used_at` | TIMESTAMP | Last usage time |
| `created_at` | TIMESTAMP | When proxy was added |
| `updated_at` | TIMESTAMP | Last update time |

---

## Running Multiple Workers

Once database mode is enabled and tasks are loaded, you can run multiple workers in parallel:

### On the same machine:
```bash
# Terminal 1
python -m seller_validator

# Terminal 2
python -m seller_validator

# Terminal 3
python -m seller_validator
```

### On different machines:
```bash
# Machine 1
python -m seller_validator

# Machine 2 (must have access to database)
python -m seller_validator

# Machine 3
python -m seller_validator
```

**How it works:**
- Each worker fetches tasks from the database using `FOR UPDATE SKIP LOCKED`
- This ensures no two workers process the same task simultaneously
- Tasks are automatically marked as `in_progress` when fetched
- Results are written back to the database atomically
- Failed tasks are automatically retried (up to `max_attempts`)

---

## Workflow Example

### Initial Setup (One-time)

1. **Enable database mode:**
   ```bash
   # Edit seller_validator/config.py
   # Set: use_database = True
   ```

2. **Migrate existing results:**
   ```bash
   python scripts/migrate_results.py
   ```

3. **Load proxies:**
   ```bash
   python scripts/load_proxies.py
   # Choose mode: r (replace)
   ```

4. **Load initial tasks:**
   ```bash
   python scripts/load_urls.py
   # Choose mode: a (add)
   ```

### Regular Operations

**Add new URLs to process:**
```bash
# Update data/urls.txt with new URLs
python scripts/load_urls.py
# Choose mode: a (add)
```

**Replace pending tasks with new batch:**
```bash
# Update data/urls.txt with new batch
python scripts/load_urls.py
# Choose mode: r (replace)
```

**Start workers:**
```bash
# Start as many workers as needed
python -m seller_validator
```

**Monitor progress:**
```bash
# Use load_urls.py to see task statistics
python scripts/load_urls.py
# Press Ctrl+C after seeing stats (before choosing mode)
```

---

## Safety Notes

- ✅ **Replace mode for URLs** is safe while workers are running
  - Only deletes `pending` tasks
  - Does not affect `in_progress` or `completed` tasks

- ⚠️ **Replace mode for proxies** clears ALL proxies
  - Use `add` mode to add new proxies while preserving existing ones

- ✅ **Database-backed queue** uses row-level locks
  - Safe for concurrent access from multiple workers
  - No risk of duplicate processing

- ✅ **Automatic retry mechanism**
  - Failed tasks are automatically retried up to `max_attempts`
  - Tasks exceeding max attempts are marked as `abandoned`

---

## Troubleshooting

### "No tasks available"
```bash
# Check if tasks exist in database
python scripts/load_urls.py
# Look at "Current task statistics"

# If no pending tasks, load some:
python scripts/load_urls.py
# Choose mode: a
```

### "No proxies found in database"
```bash
# Load proxies first:
python scripts/load_proxies.py
# Choose mode: a (or r)
```

### "Connection refused"
- Check database host/port in `config.py`
- Verify database is accessible from your machine
- Check firewall settings

### "Tasks stuck in in_progress"
This can happen if a worker crashes or is terminated unexpectedly.

**Solution:**
```bash
# Check for stuck tasks (dry run)
python scripts/recover_stuck_tasks.py

# Recover stuck tasks (default 60 minute timeout)
python scripts/recover_stuck_tasks.py --apply

# Use custom timeout if needed
python scripts/recover_stuck_tasks.py --timeout 30 --apply
```

---

## Advanced Usage

### Querying the database

You can connect to PostgreSQL to query results:

```bash
psql -h 81.30.105.134 -p 5414 -U admin -d avito_seller_checker
```

**Useful queries:**

```sql
-- Count tasks by state
SELECT state, COUNT(*) FROM tasks GROUP BY state;

-- Count results by verdict
SELECT verdict, COUNT(*) FROM results GROUP BY verdict;

-- Get recently completed tasks
SELECT url, verdict, reason, created_at
FROM results
ORDER BY created_at DESC
LIMIT 10;

-- Find tasks that failed
SELECT t.url, r.reason
FROM tasks t
JOIN results r ON t.id = r.task_id
WHERE r.verdict = 'error';

-- Proxy statistics
SELECT
  COUNT(*) FILTER (WHERE is_blocked = FALSE) as available,
  COUNT(*) FILTER (WHERE is_blocked = TRUE) as blocked,
  COUNT(*) as total
FROM proxies;
```

---

## Migration Path: File → Database

If you're currently using file mode and want to migrate to database mode:

1. **Backup your data:**
   ```bash
   cp -r data/ data_backup/
   ```

2. **Enable database mode in config.py**

3. **Migrate existing results:**
   ```bash
   python scripts/migrate_results.py
   ```

4. **Load proxies:**
   ```bash
   python scripts/load_proxies.py
   ```

5. **Load pending URLs:**
   ```bash
   python scripts/load_urls.py
   ```

6. **Test with one worker:**
   ```bash
   python -m seller_validator
   ```

7. **Scale to multiple workers once verified**

---

## Database Mode vs File Mode

| Feature | File Mode | Database Mode |
|---------|-----------|---------------|
| **Multiple workers** | ❌ No | ✅ Yes |
| **Distributed processing** | ❌ No | ✅ Yes |
| **Task persistence** | ⚠️ In-memory only | ✅ Persistent |
| **Crash recovery** | ❌ Lost tasks | ✅ Automatic |
| **Query results** | ⚠️ Parse JSON | ✅ SQL queries |
| **Setup complexity** | ✅ Simple | ⚠️ Requires DB |
| **Scalability** | ⚠️ Single machine | ✅ Horizontal |

---

For questions or issues, please refer to the main project README.
