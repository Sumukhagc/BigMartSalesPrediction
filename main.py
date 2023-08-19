from src.BigMartSalesPrediction.pipeline.dataingestionpipeline import DataIngestionPipeline
from src.BigMartSalesPrediction.pipeline.datatransformationpipeline import DataTransformationPipeline
from src.BigMartSalesPrediction.pipeline.modeltrainerpipeline import ModelTrainerPipeline
from src.BigMartSalesPrediction.pipeline.modelevaluationpipeline import ModelEvaluationPipeline
try:
    data_ingestion_pipeline=DataIngestionPipeline()
    data_ingestion_pipeline.main()
except Exception as e:
    print(e)   

try:
    data_transformation_pipeline=DataTransformationPipeline()
    data_transformation_pipeline.main()

except Exception as e:
    print(e)

try:
    model_trainer_pipeline=ModelTrainerPipeline()
    #model_trainer_pipeline.main()

except Exception as e:
    print(e)   

try:
    model_evaluation_pipeline=ModelEvaluationPipeline()
    model_evaluation_pipeline.main()

except Exception as e:
    print(e)        
