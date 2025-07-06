import umap


def get_train_test_umap(X_train, X_test, n_components=2):
    umap_model = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)

    # Fit UMAP on the training data with labels
    X_train_embedding = umap_model.fit_transform(X_train)

    # Transform the test data into the existing UMAP embedding
    X_test_embedding = umap_model.transform(X_test)

    return X_train_embedding, X_test_embedding