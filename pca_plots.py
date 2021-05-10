def perform_pca(cancer_data):

    ### Z-score standardization
    # Separate features
    X = cancer_data.iloc[:,2:].values

    # Separate diagnosis
    diagnosis = cancer_data.loc[:,['diagnosis']].values

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    ### PCA

    # Covariance matrix
    covariance_mat = PCA() # if no nr of features is defined, all of them will be used
    #print(covariance_mat)

    # Get principal components
    pcs = covariance_mat.fit_transform(X)
    # Grab only the first 2 PCs
    pca_res = pd.DataFrame(data = pcs[:, 0:2],
                           columns = ['PC1', 'PC2'])

    # Add back diagnosis column
    pca_res = pd.concat([pca_res, cancer_data[['diagnosis']]], axis = 1)

    # Make names pretty
    pca_res = pca_res.replace(to_replace='B', value='Benign')
    pca_res = pca_res.replace(to_replace='M', value='Malignant')

    # plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title('PCA', fontsize = 18)

    targets = ['Malignant', 'Benign']
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indices = pca_res['diagnosis'] == target
        ax.scatter(pca_res.loc[indices, 'PC1'], pca_res.loc[indices, 'PC2'],
                   c = color, s = 50)

    ax.legend(targets)
    ax.grid()

    return pca_res