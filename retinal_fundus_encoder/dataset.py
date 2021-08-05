import os

import pandas as pd
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EyePACSDataset(torch.utils.data.Dataset):
    """ Data source: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
    """

    def __init__(self, root_dir, csv_file=None, num_views=1):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            num_views (int): Number of views to be generated from each image.
        """
        self.root_dir = root_dir
        self.num_views = num_views
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
            self.data['normal-abnormal'] = self.data.apply(lambda row: 0 if row['level'] == 0 else 1, axis=1)
        else:
            self.data = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(self.data, pd.DataFrame):
            row = self.data.iloc[index]
            image_id = row['image']
            image = Image.open(os.path.join(self.root_dir, f"{image_id}.jpeg"))
            label = row['normal-abnormal']
        else:
            image_id = self.data[index].split('.')[0]
            image = Image.open(os.path.join(self.root_dir, self.data[index]))
            label = -1

        images = [image] * self.num_views
        labels = {'image_id': image_id, 'normal-abnormal': [label] * self.num_views}

        return images, labels


class OIAODIRDataset(torch.utils.data.Dataset):
    """ Data source: https://github.com/nkicsl/OIA-ODIR
    """

    def __init__(self, root_dir, csv_file, num_views=1):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            num_views (int): Number of views to be generated from each image.
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_id = row['image'].split('.')[0]
        image = Image.open(os.path.join(self.root_dir, f"{image_id}.jpg"))
        label = row['normal-abnormal']

        images = [image] * self.num_views
        labels = {'image_id': image_id, 'normal-abnormal': [label] * self.num_views}

        return images, labels


class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_path, label_scheme='all', abnormality_threshold=0.5):
        """
        Args:
            image_dir (string): Directory with images generated from `4_generate_patch_dataset.py`.
            label_path (string): Label file generated from `4_generate_patch_dataset.py`.
            label_scheme (string): Patch label scheme. Available options: ['position', 'abnormality', 'patient',
                                   'position-abnormality', 'position-patient', 'abnormality-patient', 'all'].
            abnormality_threshold (float): Threshold for abnormality scores to be considered as positive. Applicable
                                        when either 'abnormality', 'position-abnormality', 'abnormality-patient' or
                                        'all' is selected for `label_scheme`.
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(label_path)
        self.label_scheme = label_scheme
        self.abnormality_threshold = abnormality_threshold
        self.positions = {"tl": 0, "tr": 1, "c": 2, "bl": 3, "br": 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        pair_id = row['id']
        patient_id = int(row['left-patch'].split("_")[1])
        if 'eyepacs' not in row['left-patch']:
            # add offset to patient ID for images from OIA-ODIR dataset
            patient_id += 44352
        position = self.positions[row['left-patch'].split("_")[-1]]

        left_filename = f"{row['left-patch']}.jpg"
        left_patch = Image.open(os.path.join(self.image_dir, left_filename))
        left_abnormality = 1 if row['left-score'] >= self.abnormality_threshold else 0
        left_label = pair_id

        right_filename = f"{row['right-patch']}.jpg"
        right_patch = Image.open(os.path.join(self.image_dir, right_filename))
        right_abnormality = 1 if row['right-score'] >= self.abnormality_threshold else 0
        right_label = pair_id

        if self.label_scheme == 'position':
            # assign same label to patches from the same position
            left_label = position
            right_label = position
        elif self.label_scheme == 'abnormality':
            # assign same label to patches with the same abnormality label
            left_label = left_abnormality
            right_label = right_abnormality
        elif self.label_scheme == 'patient':
            # assigns same label to patches from the same patient
            left_label = patient_id
            right_label = patient_id
        elif self.label_scheme == 'position-abnormality':
            # assign same label to patches with the same position and abnormality label
            left_label = position + len(self.positions) if left_abnormality == 1 else position
            right_label = position + len(self.positions) if right_abnormality == 1 else position
        elif self.label_scheme == 'position-patient':
            # assign same label to patches with the same position and patient label
            left_label = pair_id
            right_label = pair_id
        elif self.label_scheme == 'abnormality-patient':
            # assign same label to patches with the same abnormality and patient label
            left_label = patient_id + len(self.data) if left_abnormality == 1 else patient_id
            right_label = patient_id + len(self.data) if right_abnormality == 1 else patient_id
        elif self.label_scheme == 'all':
            # assign same label to patches with the same position, abnormality, and patient label
            if right_abnormality != left_abnormality:
                right_label = pair_id + 1
        else:
            raise ValueError(f"Unknown label scheme '{self.label_scheme}'")

        return [left_patch, right_patch, left_patch, right_patch], [left_label, right_label]


class PreTrainMultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_views=2):
        """
        Args:
            root_dir (string): Directory with all the images.
            num_views (int): Number of views to be generated from one image.
        """
        self.root_dir = root_dir
        self.data = os.listdir(root_dir)
        self.num_views = num_views

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data[index]
        image = Image.open(os.path.join(self.root_dir, filename))

        return [image] * self.num_views


class PreTrainMultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_path, abnormality_threshold=0.5):
        """
        Args:
            image_dir (string): Directory with images generated from `4_generate_patch_dataset.py`.
            label_path (string): Label file generated from `4_generate_patch_dataset.py`.
            abnormality_threshold (float): Threshold for abnormality scores to be considered as positive.
        """
        self.image_dir = image_dir
        self.data = pd.read_csv(label_path)
        self.abnormality_threshold = abnormality_threshold
        self.positions = {"tl": 0, "tr": 1, "c": 2, "bl": 3, "br": 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        left_filename = f"{row['left-patch']}.jpg"
        left_patch = Image.open(os.path.join(self.image_dir, left_filename))
        left_abnormality = 1 if row['left-score'] >= self.abnormality_threshold else 0
        left_position = self.positions[row['left-patch'].split("_")[-1]]

        right_filename = f"{row['right-patch']}.jpg"
        right_patch = Image.open(os.path.join(self.image_dir, right_filename))
        right_abnormality = 1 if row['right-score'] >= self.abnormality_threshold else 0
        right_position = self.positions[row['right-patch'].split("_")[-1]]

        assert left_position == right_position
        labels = {
            "normal-abnormal": [left_abnormality, right_abnormality],
            "position": [left_position, right_position]
        }

        return [left_patch, right_patch], labels


class MessidorDataset(torch.utils.data.Dataset):
    """ Data source: http://www.adcis.net/en/third-party/messidor/
    """
    TASK_LABELS = {
        'retinopathy-referability': ('nonreferable', 'referable'),
        'retinopathy-normal-abnormal': ('normal', 'abnormal'),
        'macular-edema-risk': ('0 (no risk)', '1', '2')
    }

    FOLDER_NAMES = (
        'Base11', 'Base12', 'Base13', 'Base14',
        'Base21', 'Base22', 'Base23', 'Base24',
        'Base31', 'Base32', 'Base33', 'Base34'
    )

    INCONSISTENT_GRADINGS = {
        ('Base33', '20051202_55562_0400_PP.tif'),
        ('Base33', '20051205_33025_0400_PP.tif'),
        ('Base33', '20051202_55626_0400_PP.tif'),
        ('Base33', '20051202_54611_0400_PP.tif')
    }

    DR_LABEL_CORRECTIONS = {
        ('Base11', '20051020_64007_0100_PP.tif'): {'old value': 1, 'new value': 3},
        ('Base11', '20051020_63936_0100_PP.tif'): {'old value': 3, 'new value': 1},
        ('Base13', '20060523_48477_0100_PP.tif'): {'old value': 2, 'new value': 3},
        ('Base11', '20051020_63045_0100_PP.tif'): {'old value': 3, 'new value': 0},
    }

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all sub-folders from different ophthalmology departments.
        """
        self.root_dir = root_dir
        self.data = pd.DataFrame()
        for folder_name in MessidorDataset.FOLDER_NAMES:
            excel_path = os.path.join(root_dir, folder_name, f"Annotation_{folder_name}.xls")
            if not os.path.exists(excel_path):
                excel_path = os.path.join(root_dir, folder_name, f"Annotation {folder_name}.xls")
            df = pd.read_excel(excel_path)
            df["folder_name"] = folder_name

            self.data = pd.concat([self.data, df])
        self.data.reset_index(drop=True, inplace=True)

        for folder_name, image_file in self.INCONSISTENT_GRADINGS:
            conditions = (self.data['folder_name'] == folder_name) & (self.data['Image name'] == image_file)
            self.data.drop(self.data[conditions].index, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        folder_name = row['folder_name']
        image_file = row['Image name']
        dr_grade = row['Retinopathy grade']
        edema_risk = row['Risk of macular edema ']

        if (folder_name, image_file) in self.DR_LABEL_CORRECTIONS:
            assert dr_grade == self.DR_LABEL_CORRECTIONS[(folder_name, image_file)]['old value']
            dr_grade = self.DR_LABEL_CORRECTIONS[(folder_name, image_file)]['new value']

        image = Image.open(os.path.join(self.root_dir, folder_name, image_file))
        labels = {
            'retinopathy-referability': int(dr_grade > 1),
            'retinopathy-normal-abnormal': int(dr_grade >= 1),
            'macular-edema-risk': int(edema_risk)
        }

        return image, labels


class IDRiDGradingDataset(torch.utils.data.Dataset):
    """ Data source: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    """
    TASK_LABELS = {
        'retinopathy-grade': ('0 (normal)', '1', '2', '3', '4'),
        'macular-edema-risk': ('0 (no risk)', '1', '2')
    }

    def __init__(self, root_dir, csv_file):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_id = row['Image name']
        dr_grade = row['Retinopathy grade']
        edema_risk = row['Risk of macular edema ']

        image = Image.open(os.path.join(self.root_dir, f"{image_id}.jpg"))
        labels = {
            'retinopathy-grade': int(dr_grade),
            'macular-edema-risk': int(edema_risk)
        }

        return image, labels


class REFUGEGradingDataset(torch.utils.data.Dataset):
    """ Data source: https://refuge.grand-challenge.org/
    """
    TASK_LABELS = {
        'glaucoma-grade': ('0 (normal)', '1')
    }

    def __init__(self, root_dir, csv_file):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_id = row['image_id']
        label = row['glaucoma_label']

        image = Image.open(os.path.join(self.root_dir, f"{image_id}.jpg"))

        return image, label


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the segmentation masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_filename, segmentation_filename = self.data[index]

        image = Image.open(os.path.join(self.image_dir, image_filename))
        segmentation_mask = Image.open(os.path.join(self.mask_dir, segmentation_filename))

        return image, segmentation_mask


class DRIVEDataset(SegmentationDataset):
    """ Data source: https://drive.grand-challenge.org/DRIVE/
    """
    TASK_LABELS = {
        'retinal-vessel': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.tif).
            mask_dir (string): Directory with all the segmentation masks (.gif).
        """
        super().__init__(image_dir, mask_dir)

        self.data = []
        for filename in os.listdir(image_dir):
            image_id, extension = filename.split('.')
            if extension == 'tif':
                self.data.append((filename, f"{image_id.split('_')[0]}_manual1.gif"))


class STAREDataset(SegmentationDataset):
    """ Data source: https://cecas.clemson.edu/~ahoover/stare/probing/index.html
    """
    TASK_LABELS = {
        'retinal-vessel': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.ppm).
            mask_dir (string): Directory with all the segmentation masks (.ah.ppm).
        """
        super().__init__(image_dir, mask_dir)

        self.data = []
        for filename in os.listdir(image_dir):
            if len(filename.split('.')) == 2:
                image_id = filename.split('.')[0]
                self.data.append((f"{image_id}.ppm", f"{image_id}.ah.ppm"))


class CHASEDB1Dataset(SegmentationDataset):
    """ Data source: https://blogs.kingston.ac.uk/retinal/chasedb1/
    """
    TASK_LABELS = {
        'retinal-vessel': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.jpg).
            mask_dir (string): Directory with all the segmentation masks (.png).
        """
        super().__init__(image_dir, mask_dir)

        self.data = []
        for filename in os.listdir(image_dir):
            image_id, extension = filename.split('.')
            if extension == 'jpg':
                self.data.append((filename, f"{image_id}_1stHO.png"))


class REFUGESegmentationDataset(torch.utils.data.Dataset):
    """ Data source: https://refuge.grand-challenge.org/
    """
    TASK_LABELS = {
        'optic disc': ('absent', 'present'),
        'optic cup': ('absent', 'present'),
    }

    def __init__(self, image_dir, mask_dir):
        """
        Args:
            image_dir (string): Directory with all the images (.jpg).
            mask_dir (string): Directory with all the segmentation masks (.bmp).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.data = []
        for filename in os.listdir(image_dir):
            image_id = filename.split('.')[0]
            self.data.append((filename, f"{image_id}.bmp"))

        self.optic_disc_masks = dict()
        self.optic_cup_masks = dict()
        for _, filename in self.data:
            segmentation_mask = Image.open(os.path.join(self.mask_dir, filename))
            self.optic_disc_masks[filename] = segmentation_mask.point(lambda p: 255 if p != 255 else 0)
            self.optic_cup_masks[filename] = segmentation_mask.point(lambda p: 255 if p == 0 else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_filename, segmentation_filename = self.data[index]

        image = Image.open(os.path.join(self.image_dir, image_filename))
        optic_disc_mask = self.optic_disc_masks[segmentation_filename]
        optic_cup_mask = self.optic_cup_masks[segmentation_filename]

        return image, {"optic disc": optic_disc_mask, "optic cup": optic_cup_mask}
