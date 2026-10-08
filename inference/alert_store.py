import json
import sqlite3
import threading
from pathlib import Path


class AlertStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.db:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, truck_id TEXT, class INTEGER, label TEXT, source TEXT, probs_json TEXT, synced INTEGER DEFAULT 0)")
            self.db.execute("CREATE TABLE IF NOT EXISTS windows (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, truck_id TEXT, features_json TEXT, probs_json TEXT, final_class INTEGER, synced INTEGER DEFAULT 0)")
            self.db.execute("CREATE TABLE IF NOT EXISTS door_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, truck_id TEXT, seq INTEGER, event TEXT, synced INTEGER DEFAULT 0)")

    def window(self, ts, truck_id, features, probs, final_class):
        with self.lock, self.db:
            self.db.execute("INSERT INTO windows(ts, truck_id, features_json, probs_json, final_class) VALUES (?, ?, ?, ?, ?)", (ts, truck_id, json.dumps([float(x) for x in features]), json.dumps([float(x) for x in probs]), final_class))

    def alert(self, ts, truck_id, class_id, label, source, probs):
        with self.lock, self.db:
            cur = self.db.execute("INSERT INTO alerts(ts, truck_id, class, label, source, probs_json) VALUES (?, ?, ?, ?, ?, ?)", (ts, truck_id, class_id, label, source, json.dumps([float(x) for x in probs])))
            return cur.lastrowid

    def door_event(self, ts, truck_id, seq, event):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO door_events(ts, truck_id, seq, event) VALUES (?, ?, ?, ?)",
                (ts, truck_id, seq, event),
            )

    def unsynced_alerts(self, limit=100):
        with self.lock:
            rows = self.db.execute("SELECT id, ts, truck_id, class, label, source, probs_json FROM alerts WHERE synced=0 ORDER BY id LIMIT ?", (limit,)).fetchall()
        return rows

    def mark_synced(self, ids):
        if not ids:
            return
        with self.lock, self.db:
            self.db.executemany("UPDATE alerts SET synced=1 WHERE id=?", ((i,) for i in ids))

    def unsynced_door_events(self, limit=100):
        with self.lock:
            return self.db.execute(
                "SELECT id, ts, truck_id, seq, event FROM door_events "
                "WHERE synced=0 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()

    def mark_door_events_synced(self, ids):
        if not ids:
            return
        with self.lock, self.db:
            self.db.executemany(
                "UPDATE door_events SET synced=1 WHERE id=?",
                ((event_id,) for event_id in ids),
            )

    def close(self):
        with self.lock:
            self.db.close()
