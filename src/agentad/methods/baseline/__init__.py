"""Training-free anomaly-detection baselines.

Every module provides a frozen config dataclass plus a ``score(series,
config)`` function mapping ``[batch, time, feature]`` tensors to ``[batch,
time]`` anomaly scores; whatever fitting a method does — clustering,
robust covariance, the SVM dual — happens inside the call.
"""

from .cluster_local_outlier_factor import (
    ClusterLocalOutlierFactorConfig,
    score as cluster_local_outlier_factor_score,
)
from .connectivity_outlier_factor import (
    ConnectivityOutlierFactorConfig,
    score as connectivity_outlier_factor_score,
)
from .copula import CopulaConfig, score as copula_score
from .extended_isolation_forest import (
    ExtendedIsolationForestConfig,
    score as extended_isolation_forest_score,
)
from .fourier import FourierConfig, score as fourier_score
from .histogram import HistogramConfig, score as histogram_score
from .isolation_forest import (
    IsolationForestConfig,
    score as isolation_forest_score,
)
from .kmeans_distance import KMeansDistanceConfig, score as kmeans_distance_score
from .local_outlier_factor import (
    LocalOutlierFactorConfig,
    score as local_outlier_factor_score,
)
from .local_polynomial import LocalPolynomialConfig, score as local_polynomial_score
from .matrix_profile import MatrixProfileConfig, score as matrix_profile_score
from .minimum_covariance_determinant import (
    MinimumCovarianceDeterminantConfig,
    score as minimum_covariance_determinant_score,
)
from .nearest_neighbors import (
    NearestNeighborsConfig,
    score as nearest_neighbors_score,
)
from .one_class_svm import OneClassSVMConfig, score as one_class_svm_score
from .principal_component import (
    PrincipalComponentConfig,
    score as principal_component_score,
)
from .robust_pca import RobustPCAConfig, score as robust_pca_score
from .spectral_residual import SpectralResidualConfig, score as spectral_residual_score
from .statistical_features import (
    StatisticalFeaturesConfig,
    score as statistical_features_score,
)
from .streaming_matrix_profile import (
    StreamingMatrixProfileConfig,
    score as streaming_matrix_profile_score,
)

BASELINES = {
    "ClusterLocalOutlierFactor": cluster_local_outlier_factor_score,
    "ConnectivityOutlierFactor": connectivity_outlier_factor_score,
    "Copula": copula_score,
    "ExtendedIsolationForest": extended_isolation_forest_score,
    "Fourier": fourier_score,
    "Histogram": histogram_score,
    "IsolationForest": isolation_forest_score,
    "KMeansDistance": kmeans_distance_score,
    "LocalOutlierFactor": local_outlier_factor_score,
    "LocalPolynomial": local_polynomial_score,
    "MatrixProfile": matrix_profile_score,
    "MinimumCovarianceDeterminant": minimum_covariance_determinant_score,
    "NearestNeighbors": nearest_neighbors_score,
    "OneClassSVM": one_class_svm_score,
    "PrincipalComponent": principal_component_score,
    "RobustPCA": robust_pca_score,
    "SpectralResidual": spectral_residual_score,
    "StatisticalFeatures": statistical_features_score,
    "StreamingMatrixProfile": streaming_matrix_profile_score,
}

__all__ = [
    "BASELINES",
    "ClusterLocalOutlierFactorConfig",
    "ConnectivityOutlierFactorConfig",
    "CopulaConfig",
    "ExtendedIsolationForestConfig",
    "FourierConfig",
    "HistogramConfig",
    "IsolationForestConfig",
    "KMeansDistanceConfig",
    "LocalOutlierFactorConfig",
    "LocalPolynomialConfig",
    "MatrixProfileConfig",
    "MinimumCovarianceDeterminantConfig",
    "NearestNeighborsConfig",
    "OneClassSVMConfig",
    "PrincipalComponentConfig",
    "RobustPCAConfig",
    "SpectralResidualConfig",
    "StatisticalFeaturesConfig",
    "StreamingMatrixProfileConfig",
    "cluster_local_outlier_factor_score",
    "connectivity_outlier_factor_score",
    "copula_score",
    "extended_isolation_forest_score",
    "fourier_score",
    "histogram_score",
    "isolation_forest_score",
    "kmeans_distance_score",
    "local_outlier_factor_score",
    "local_polynomial_score",
    "matrix_profile_score",
    "minimum_covariance_determinant_score",
    "nearest_neighbors_score",
    "one_class_svm_score",
    "principal_component_score",
    "robust_pca_score",
    "spectral_residual_score",
    "statistical_features_score",
    "streaming_matrix_profile_score",
]
