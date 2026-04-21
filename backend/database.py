from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import text

DATABASE_URL = "sqlite:///./nocode_cv.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

def create_db():
    SQLModel.metadata.create_all(engine)
    with engine.connect() as conn:
        # ── Annotation migrations ──────────────────────────────────────────
        for col, default in [
            ("shape_type",  "'bbox'"),
            ("points_json", "'[]'"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE annotation ADD COLUMN {col} TEXT DEFAULT {default}"))
                conn.commit()
            except Exception:
                pass

        # ── Image metadata migrations ──────────────────────────────────────
        for col, typ, default in [
            ("md5_hash",    "TEXT",    "''"),
            ("width",       "INTEGER", "0"),
            ("height",      "INTEGER", "0"),
            ("channels",    "INTEGER", "3"),
            ("color_space", "TEXT",    "'RGB'"),
            ("is_corrupt",  "INTEGER", "0"),
            ("file_size",   "INTEGER", "0"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE image ADD COLUMN {col} {typ} DEFAULT {default}"))
                conn.commit()
            except Exception:
                pass

        # ── TrainingRun extended fields ────────────────────────────────────
        for col, typ, default in [
            ("run_dir",         "TEXT", "''"),
            ("onnx_path",       "TEXT", "''"),
            ("aug_config_json", "TEXT", "'{}'"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE trainingrun ADD COLUMN {col} {typ} DEFAULT {default}"))
                conn.commit()
            except Exception:
                pass

def get_session():
    with Session(engine) as session:
        yield session
