import numpy as np 

def mode_based_variance(predictions: np.array) -> np.array:
    """Compute the variance of a set of stocastically produced predictions. 
    
    Source papers: 
        Kwon, Y., Won, J. H., Kim, B. J., & Paik, M. C. (2020). Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation. Computational Statistics & Data Analysis, 142, 106816.
        Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision?. Advances in neural information processing systems, 30.
        Bao, W., Yu, Q., & Kong, Y. (2020, October). Uncertainty-based traffic accident anticipation with spatio-temporal relational learning. In Proceedings of the 28th ACM International Conference on Multimedia (pp. 2682-2690).
    
    Latex formula:
        U_{ept} =  \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - \bar{p}) (\hat{p}_i - \bar{p})^ \top
    
    Args:
        predictions (np.array): shape of (n,m,k). n=number of stocastic predictions, m=number of instances (e.g., the size of the test dataset), k=number of classes 
    Returns:
        np.array: shape of (m,k) with the average variance compared to the mean for every instance.
    """
    # init list to save uncertainty values
    epistemic_uc = []
    aleatoric_uc = []
    
    # loop through the predictions
    for i in range(predictions.shape[1]): # assumes that [1] is the number of instances
        prediction_mean = np.mean(predictions[:,i,:], axis=0) # mean of the n stocastic predictions
        p_hat = np.squeeze(predictions[:,i,:])
        aleatoric_instance =  np.diag(prediction_mean) - p_hat.T.dot(p_hat)/ p_hat.shape[1]
        aleatoric_uc.append(aleatoric_instance.diagonal())
        
        prediction_distance = np.subtract(predictions[:,i,:], prediction_mean) # distance for each prediction to the mean
        epistemic_instance = (prediction_distance.T.dot(prediction_distance)/prediction_distance.shape[0]).diagonal() # diagonal of the covariance matrix (e.g., the variance per class k)
        epistemic_uc.append(epistemic_instance)
        
    
    return np.array(epistemic_uc), np.std(predictions, axis=0)
