import os
import csv
from app import create_app, db
from app.models import ImageData, GroundTruth

def create_tables():
    """Create the database tables."""
    db.create_all()

def populate_training_data(base_folder):
    """Populate the training data."""
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    csv_file_path = os.path.join(folder_path, file_name)
                    with open(csv_file_path, 'r') as file:
                        csv_reader = csv.DictReader(file, delimiter=';')
                        for row in csv_reader:
                            data = ImageData(
                                folder_name=folder_name,
                                filename=row['Filename'],
                                width=int(row['Width']),
                                height=int(row['Height']),
                                roi_x1=int(row['Roi.X1']),
                                roi_y1=int(row['Roi.Y1']),
                                roi_x2=int(row['Roi.X2']),
                                roi_y2=int(row['Roi.Y2']),
                                class_id=int(row['ClassId'])
                            )
                            db.session.add(data)
            db.session.commit()

def populate_test_data(test_folder):
    """Populate the test data."""
    csv_file_path = os.path.join(test_folder, 'GT-test.csv')
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=';')
        for row in csv_reader:
            data = ImageData(
                folder_name='test',
                filename=row['Filename'],
                width=int(row['Width']),
                height=int(row['Height']),
                roi_x1=int(row['Roi.X1']),
                roi_y1=int(row['Roi.Y1']),
                roi_x2=int(row['Roi.X2']),
                roi_y2=int(row['Roi.Y2']),
                class_id=None
            )
            db.session.add(data)
    db.session.commit()

def populate_ground_truth(ground_truth_csv):
    """Populate the ground truth data."""
    with open(ground_truth_csv, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=';')
        for row in csv_reader:
            ground_truth = GroundTruth(
                filename=row['Filename'],
                ground_truth_class_id=int(row['ClassId'])
            )
            db.session.add(ground_truth)
    db.session.commit()


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        create_tables()
        populate_training_data('data/raw/Final_Training_Images/')
        populate_test_data('data/raw/Final_Test_Images/')
        populate_ground_truth('data/raw/Final_Test_Images/GT-groundtruth.csv')
