"""
Enroll all students from seed folders.
Expected folders:
  static/captures/seed/student_1_samples/
  static/captures/seed/student_2_samples/
  static/captures/seed/student_3_samples/
  static/captures/seed/student_4_samples/
"""

from datetime import datetime
from pathlib import Path
import shutil

from flask import Flask

from database import Individual, db, init_db
from recognizer import EMBEDDINGS_PATH, FaceRecognizer

BASE_DIR = Path(__file__).resolve().parent
SEED_DIR = BASE_DIR / "static" / "captures" / "seed"

STUDENTS = [
    {
        "full_name": "Demo Student 1",
        "id_number": "DEMO-1001",
        "role": "Student",
        "clearance_status": True,
        "samples_dir": "student_1_samples",
        "photo_file": "student_1.jpg",
    },
    {
        "full_name": "Demo Student 2",
        "id_number": "DEMO-1002",
        "role": "Student",
        "clearance_status": True,
        "samples_dir": "student_2_samples",
        "photo_file": "student_2.jpg",
    },
    {
        "full_name": "Demo Student 3",
        "id_number": "DEMO-1003",
        "role": "Student",
        "clearance_status": True,
        "samples_dir": "student_3_samples",
        "photo_file": "student_3.jpg",
    },
    {
        "full_name": "Demo Student 4",
        "id_number": "DEMO-1004",
        "role": "Student",
        "clearance_status": True,
        "samples_dir": "student_4_samples",
        "photo_file": "student_4.jpg",
    },
]

def build_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{BASE_DIR / 'surveillance.db'}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    init_db(app)
    return app

def ensure_seed_assets():
    missing = []
    for student in STUDENTS:
        sample_dir = SEED_DIR / student["samples_dir"]
        if not sample_dir.exists():
            missing.append(str(sample_dir))
            continue
        images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.jpeg")) + list(sample_dir.glob("*.png"))
        if not images:
            missing.append(str(sample_dir))
    if missing:
        raise FileNotFoundError(
            "Missing required seed folders:\n- " + "\n- ".join(missing)
        )

def sync_profile_photos():
    """Use the first sample image as the profile photo."""
    for student in STUDENTS:
        sample_dir = SEED_DIR / student["samples_dir"]
        profile_photo = SEED_DIR / student["photo_file"]
        images = sorted(
            list(sample_dir.glob("*.jpg"))
            + list(sample_dir.glob("*.jpeg"))
            + list(sample_dir.glob("*.png"))
        )
        if images:
            shutil.copy2(images[0], profile_photo)

def enroll():
    ensure_seed_assets()
    sync_profile_photos()
    app = build_app()

    with app.app_context():
        current_max = db.session.query(db.func.max(Individual.face_encoding_label)).scalar() or 0
        next_label = current_max + 1

        for student in STUDENTS:
            person = Individual.query.filter_by(id_number=student["id_number"]).first()
            if person is None:
                person = Individual(
                    id_number=student["id_number"],
                    face_encoding_label=next_label,
                    registered_at=datetime.now(),
                )
                db.session.add(person)
                next_label += 1

            person.full_name = student["full_name"]
            person.role = student["role"]
            person.clearance_status = student["clearance_status"]
            person.photo_path = f"static/captures/seed/{student['photo_file']}"

        db.session.commit()

        recognizer = FaceRecognizer()
        ok = recognizer.train_from_database(app)
        if not ok:
            raise RuntimeError("Training failed. Check sample folders and images.")

        print("\n✅ Enrolled students:")
        for student in STUDENTS:
            person = Individual.query.filter_by(id_number=student["id_number"]).first()
            print(f"  - {person.full_name} ({person.id_number}) label={person.face_encoding_label}")
        print(f"Embeddings saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    enroll()