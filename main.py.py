import os
from src.dataset import load_data, get_train_test_splits
from src.model import build_svm_pipeline, train_svm
from src.evaluate import evaluate_model

def main():
    # Setup dataset path (Change dataset path as needed)
    DATA_DIR = "/kaggle/input/competitions/dogs-vs-cats/train" 
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "./data" # Fallback local directory
    
    # 1. Load Data (Limit samples to 5000 for fast CPU training, adjust if using GPU/High RAM)
    X, y = load_data(data_dir=DATA_DIR, img_size=(64, 64), max_samples=5000)
    
    # 2. Split Data (80% Train, 10% Val, 10% Test)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_splits(X, y)
    
    # 3. Build & Train SVM Pipeline (PCA + SVM)
    pipeline = build_svm_pipeline(n_components=100, kernel='rbf', C=10.0)
    trained_model = train_svm(pipeline, X_train, y_train)
    
    # 4. Evaluate on Validation and Test Set
    print("\nEvaluating on Validation Set:")
    evaluate_model(trained_model, X_val, y_val)
    
    print("\nEvaluating on Test Set:")
    evaluate_model(trained_model, X_test, y_test)

if __name__ == "__main__":
    main()