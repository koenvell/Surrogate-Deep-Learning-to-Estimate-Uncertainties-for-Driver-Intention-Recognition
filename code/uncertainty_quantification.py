import numpy as np 
from typing import Tuple

def uncertainty_quantification(predictions: np.array) -> Tuple[np.array, np.array, np.array]: 
    """Compute the variance of a set of stocastically produced predictions. 
    
    Source papers: 
        Kwon, Y., Won, J. H., Kim, B. J., & Paik, M. C. (2020). Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation. Computational Statistics & Data Analysis, 142, 106816.
        Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision?. Advances in neural information processing systems, 30.
        Bao, W., Yu, Q., & Kong, Y. (2020, October). Uncertainty-based traffic accident anticipation with spatio-temporal relational learning. In Proceedings of the 28th ACM International Conference on Multimedia (pp. 2682-2690).
   
    Args:
        predictions (np.array): shape of (n,m,k). n=number of stocastic predictions, m=number of instances (e.g., the size of the test dataset), k=number of classes 
    Returns:
        tuple: consists of the aleatoric, epistemic and predictive uncertainy in np.array format with a shape (m,k)
    """
    # init list to save uncertainty values
    aleatoric_list, epistemic_list, upred_list = [], [], []

    # loop through the predictions
    for i in range(predictions.shape[1]):
        p_hat = np.squeeze(predictions[:,i,:]) # number of stochastic predictions
        prediction = np.mean(p_hat, axis=0) # average prediction shape

        aleatoric = (np.diag(prediction) - p_hat.T.dot(p_hat)/p_hat.shape[0]).diagonal() 
        distance = p_hat - prediction # distance between the mean and the predictions
        epistemic = (distance.T.dot(distance)/distance.shape[0]).diagonal()
        u_pred = aleatoric + epistemic 
        # store values in a list
        aleatoric_list.append(aleatoric), epistemic_list.append(epistemic)
        upred_list.append(u_pred)
    return np.asarray(aleatoric_list), np.asarray(epistemic_list), np.asarray(upred_list)
