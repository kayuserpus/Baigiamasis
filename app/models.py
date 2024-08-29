from app import db

class ImageData(db.Model):
    __tablename__ = 'image_data'

    id = db.Column(db.Integer, primary_key=True)
    folder_name = db.Column(db.String(50), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    roi_x1 = db.Column(db.Integer, nullable=False)
    roi_y1 = db.Column(db.Integer, nullable=False)
    roi_x2 = db.Column(db.Integer, nullable=False)
    roi_y2 = db.Column(db.Integer, nullable=False)
    class_id = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return f"<ImageData {self.filename} - Class {self.class_id}>"

class GroundTruth(db.Model):
    __tablename__ = 'ground_truth'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    ground_truth_class_id = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<GroundTruth {self.filename} - Class {self.ground_truth_class_id}>"
