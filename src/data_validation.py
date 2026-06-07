import os
import pandas as pd
import logging

class DataValidation:
    def __init__(self, config: dict):
        self.config = config
        self.expected_columns = {
            "credit_score": ["int64", "float64"],
            "age": ["int64", "float64"],
            "tenure": ["int64", "float64"],
            "balance": ["int64", "float64"],
            "num_of_products": ["int64", "float64"],
            "has_credit_card": ["int64", "float64"],
            "is_active_member": ["int64", "float64"],
            "estimated_salary": ["int64", "float64"],
            "churn": ["int64", "float64"]
        }

    def validate_all_columns(self) -> bool:
        """Validates that all expected columns are present and have correct types."""
        try:
            logger = logging.getLogger("MLPipeline")
            logger.info("Starting data validation process...")
            
            raw_data_path = self.config["data"]["raw_data_path"]
            if not os.path.exists(raw_data_path):
                logger.error(f"Raw data file not found at {raw_data_path}")
                return False
                
            df = pd.read_csv(raw_data_path)
            all_cols = list(df.columns)
            
            validation_status = True
            error_msgs = []
            
            # Check presence of columns
            for col, expected_types in self.expected_columns.items():
                if col not in all_cols:
                    validation_status = False
                    error_msgs.append(f"Column '{col}' is missing.")
                else:
                    col_type = str(df[col].dtype)
                    if not any(expected_type in col_type for expected_type in expected_types):
                        validation_status = False
                        error_msgs.append(f"Column '{col}' type '{col_type}' is not matching expected types {expected_types}.")
            
            # Write status to artifacts
            artifacts_dir = self.config["artifacts_dir"]
            os.makedirs(artifacts_dir, exist_ok=True)
            status_file_path = os.path.join(artifacts_dir, "validation_status.txt")
            
            with open(status_file_path, "w") as f:
                if validation_status:
                    f.write("Validation Status: SUCCESS\n")
                    logger.info("Data validation completed successfully. Schema matches.")
                else:
                    f.write("Validation Status: FAILED\n")
                    f.write("\n".join(error_msgs))
                    logger.error(f"Data validation failed: {', '.join(error_msgs)}")
                    
            return validation_status
            
        except Exception as e:
            logger.error(f"Data validation failed due to error: {e}")
            raise e
