from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_svm_pipeline(n_components=100, kernel='rbf', C=1.0, gamma='scale'):
    """
    Builds a Scikit-Learn Pipeline with Scaling, PCA, and SVM.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42))
    ])
    return pipeline

def train_svm(pipeline, X_train, y_train):
    """
    Trains the SVM Pipeline.
    """
    print("Training SVM Model...")
    pipeline.fit(X_train, y_train)
    print("Training Complete!")
    return pipeline