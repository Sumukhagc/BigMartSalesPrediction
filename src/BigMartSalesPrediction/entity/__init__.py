from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path :Path
    x_train_path:Path
    x_test_path:Path
    y_train_path:Path
    y_test_path:Path
    scale_path:Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_path :Path
    x_train_path:Path
    y_train_path:Path

@dataclass
class ModelEvaluationConfig:
    root_dir:Path
    model_path:Path
    x_test_path:Path
    y_test_path:Path    
    evaluation_path:Path

