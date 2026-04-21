"""
One-time import script: loads a Roboflow YOLO dataset into the NoCode CV database.

Usage:
    cd backend
    python import_dataset.py --dataset ../Dataset --project "Smoke Detection"

The dataset folder must contain:
    data.yaml  (with nc and names)
    train/images/*.jpg  + train/labels/*.txt
    valid/images/*.jpg  + valid/labels/*.txt   (optional)
    test/images/*.jpg   + test/labels/*.txt    (optional)
"""
import argparse, json, os, shutil, uuid, yaml
from sqlmodel import Session, create_engine, select
from database import DATABASE_URL
from models import Project, Image, Annotation

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def import_split(session, project_id, img_dir, lbl_dir):
    if not os.path.isdir(img_dir):
        return 0
    count = 0
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue
        src = os.path.join(img_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        unique = f"{uuid.uuid4().hex}{ext}"
        shutil.copy2(src, os.path.join(UPLOAD_DIR, unique))
        img_rec = Image(project_id=project_id, filename=unique, original_name=fname)
        session.add(img_rec)
        session.flush()  # get img_rec.id

        lbl_file = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if os.path.isfile(lbl_file):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    ann = Annotation(
                        image_id=img_rec.id, class_id=cid, shape_type="bbox",
                        x_center=xc, y_center=yc, width=w, height=h,
                        points_json="[]",
                    )
                    session.add(ann)
        count += 1
    session.commit()
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset folder (contains data.yaml)")
    ap.add_argument("--project",  default=None,  help="Project name (defaults to dataset name from yaml)")
    args = ap.parse_args()

    dataset = os.path.abspath(args.dataset)
    yaml_path = os.path.join(dataset, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise SystemExit(f"data.yaml not found in {dataset}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    classes = cfg.get("names", [])
    project_name = args.project or cfg.get("project", os.path.basename(dataset))

    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    with Session(engine) as session:
        proj = Project(name=project_name, description=f"Imported from {dataset}", classes_json=json.dumps(classes))
        session.add(proj)
        session.commit()
        session.refresh(proj)
        print(f"Created project '{project_name}' (id={proj.id}) with classes: {classes}")

        total = 0
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(dataset, split, "images")
            lbl_dir = os.path.join(dataset, split, "labels")
            n = import_split(session, proj.id, img_dir, lbl_dir)
            if n:
                print(f"  {split}: {n} images imported")
            total += n

        print(f"\nDone! {total} images imported into project id={proj.id}")
        print(f"You can now open the UI and train project '{project_name}'")


if __name__ == "__main__":
    main()
