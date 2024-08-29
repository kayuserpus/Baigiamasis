# project_root/app/routes.py

from flask import Blueprint, jsonify
from app import db
from app.models import ImageData, GroundTruth

bp = Blueprint('main', __name__)

@bp.route('/data', methods=['GET'])
def get_all_data():
    data = ImageData.query.all()
    return jsonify([{
        'id': d.id,
        'folder_name': d.folder_name,
        'filename': d.filename,
        'width': d.width,
        'height': d.height,
        'roi_x1': d.roi_x1,
        'roi_y1': d.roi_y1,
        'roi_x2': d.roi_x2,
        'roi_y2': d.roi_y2,
        'class_id': d.class_id
    } for d in data])

@bp.route('/groundtruth', methods=['GET'])
def get_ground_truth():
    ground_truth = GroundTruth.query.all()
    return jsonify([{
        'id': g.id,
        'filename': g.filename,
        'ground_truth_class_id': g.ground_truth_class_id
    } for g in ground_truth])

@bp.route('/compare', methods=['GET'])
def compare_predictions_to_ground_truth():
    results = db.session.query(ImageData, GroundTruth).filter(
        ImageData.filename == GroundTruth.filename
    ).all()

    comparison = [{
        'filename': image.filename,
        'predicted_class_id': image.class_id,
        'ground_truth_class_id': truth.ground_truth_class_id,
        'match': image.class_id == truth.ground_truth_class_id
    } for image, truth in results]

    return jsonify(comparison)
