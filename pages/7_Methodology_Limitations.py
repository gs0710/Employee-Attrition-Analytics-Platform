"""
7_Methodology_Limitations.py - Academic and theoretical deep-dive into algorithms, metrics, and ethical guidelines.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import FAVICON_PATH, FLATICON_CDN_CSS

st.set_page_config(page_title="Methodology & Limitations | Workforce Segmentation", page_icon=str(FAVICON_PATH), layout="wide")
st.markdown(FLATICON_CDN_CSS, unsafe_allow_html=True)

st.markdown('<h2><i class="fi fi-sr-book-alt header-icon"></i>Methodology, Algorithm Mechanics & Ethical AI</h2>', unsafe_allow_html=True)
st.markdown("Academic documentation of machine learning theory, distance metrics, dimensionality reduction, and ethical guardrails.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Unsupervised Learning in HR",
    "K-Means & Hierarchical Theory",
    "PCA Mathematical Formulation",
    "Metric Interpretation Guide",
    "Ethical Guardrails & Responsible AI",
])

with tab1:
    st.markdown('<h4><i class="fi fi-sr-users-alt card-icon"></i>1. Why Unsupervised Learning for Workforce Segmentation?</h4>', unsafe_allow_html=True)
    st.markdown("""
    Workforce analytics frequently suffers from subjective organizational categorization (e.g., rigid grade levels or anecdotal manager perceptions).
    **Unsupervised Machine Learning** is uniquely suited for employee segmentation because:
    
    1. **Absence of Ground-Truth Target Labels:** Employees do not arrive with an objective, pre-assigned "segment" label. Supervised learning requires historical target labels which embed past human biases.
    2. **Latent Archetype Discovery:** Unsupervised algorithms analyze high-dimensional feature distributions simultaneously, uncovering multidimensional patterns that human analysts cannot perceive in spreadsheets.
    3. **Holistic Employee Context:** By simultaneously processing tenure, compensation, work-life balance, overtime, and satisfaction, the model groups employees by holistic workplace experience rather than single-dimensional salary or department silos.
    """)

with tab2:
    st.markdown('<h4><i class="fi fi-sr-bolt card-icon"></i>2. Mathematical Formulation: K-Means vs. Ward Hierarchical</h4>', unsafe_allow_html=True)
    
    st.markdown("### A. K-Means Algorithm Mechanics")
    st.markdown("""
    K-Means partitions $N$ observations into $K$ disjoint clusters $S = \\{S_1, S_2, \\dots, S_K\\}$ by minimizing the within-cluster sum of squares (**Inertia**):
    
    $$\\arg\\min_S \\sum_{i=1}^K \\sum_{x \\in S_i} \\| x - \\mu_i \\|^2$$
    
    where $\\mu_i$ is the geometric centroid of cluster $S_i$.
    - **Initialization:** Uses **k-means++** initialization to seed centroids far apart, preventing suboptimal local minima.
    - **Optimization:** Alternates between assignment (allocating points to closest centroid via Euclidean metric) and centroid updating until convergence.
    - **Geometric Assumption:** Assumes spherical, convex clusters of approximately equal variance.
    """)

    st.markdown("---")
    st.markdown("### B. Agglomerative Hierarchical Clustering (Ward's Linkage)")
    st.markdown("""
    Agglomerative clustering initiates with each employee as an individual singleton cluster ($N$ clusters) and iteratively merges the pair of clusters that minimizes the increase in total within-cluster variance:
    
    $$\\Delta \\text{ESS}_{AB} = \\frac{n_A n_B}{n_A + n_B} \\| \\mu_A - \\mu_B \\|^2$$
    
    where $n_A, n_B$ are sample sizes and $\\mu_A, \\mu_B$ are cluster means.
    - **Advantage:** Produces a deterministic, nested hierarchical tree (dendrogram) showing structural sub-group relationships.
    - **Constraint:** Ward linkage strictly requires Euclidean distance; applying non-Euclidean metrics invalidates variance minimization.
    """)

with tab3:
    st.markdown('<h4><i class="fi fi-sr-chart-histogram card-icon"></i>3. Principal Component Analysis (PCA) Theory</h4>', unsafe_allow_html=True)
    st.markdown("""
    PCA transforms a set of $P$ correlated features into a smaller set of $K$ orthogonal (uncorrelated) linear combinations called **Principal Components**.
    
    ### Mathematical Procedure:
    1. **Standardization:** Zero-centers each feature and scales to unit variance ($z = \\frac{x - \\mu}{\\sigma}$).
    2. **Covariance Matrix:** Computes the sample covariance matrix $\\Sigma = \\frac{1}{N-1} X^T X$.
    3. **Eigendecomposition:** Solves for eigenvalues $\\lambda_i$ and eigenvectors $v_i$:
       $$\\Sigma v_i = \\lambda_i v_i$$
    4. **Variance Contribution:** The eigenvalue $\\lambda_i$ represents the variance explained by the $i$-th principal component. The explained variance ratio is:
       $$\\text{Ratio}_i = \\frac{\\lambda_i}{\\sum_{j=1}^P \\lambda_j}$$
    
    ### Why 2D Projections Incur Information Loss:
    - In our workforce dataset, PC1 explains **26.5%** and PC2 explains **11.1%** (total 37.6%).
    - The remaining **62.4%** of variance resides in higher-order dimensions.
    - **Critical Principle:** Two employees appearing adjacent on a 2D PCA scatter plot may still differ significantly along unprojected dimensions. Therefore, clustering models must be trained on the full standardized feature matrix, not the 2D projection.
    """)

with tab4:
    st.markdown('<h4><i class="fi fi-sr-scale-comparison-alt card-icon"></i>4. Clustering Evaluation Metrics Interpretation Guide</h4>', unsafe_allow_html=True)
    st.markdown("""
    | Metric | Mathematical Range | Optimal Direction | Interpretation in HR Context |
    | :--- | :---: | :---: | :--- |
    | **Inertia (Elbow)** | $[0, \\infty)$ | Lower (Seek "Elbow") | Measures compactness. Decreases monotonically as $k$ increases; look for the point of diminishing returns. |
    | **Silhouette Score** | $[-1, +1]$ | Higher | Evaluates how well-separated clusters are. $>0.10$ in high-dimensional mixed data indicates meaningful density separation. |
    | **Davies-Bouldin Index** | $[0, \\infty)$ | Lower | Ratio of within-cluster scatter to between-cluster separation. Lower scores indicate tighter, better-separated clusters. |
    | **Calinski-Harabasz Index** | $[0, \\infty)$ | Higher | Ratio of between-cluster variance to within-cluster variance. Higher scores indicate distinct cluster definitions. |
    | **Adjusted Rand Index (ARI)** | $[-1, +1]$ | Higher | Measures agreement between two independent partitionings (e.g., K-Means vs. Hierarchical) corrected for chance. |
    """)

with tab5:
    st.markdown('<h4><i class="fi fi-sr-shield-check card-icon"></i>5. Ethical AI & Responsible Workforce Analytics</h4>', unsafe_allow_html=True)
    st.markdown("""
    ### Ethical Mandate & Legal Compliance
    Applying artificial intelligence to human workforce data requires strict adherence to ethical principles, algorithmic fairness, and non-discriminatory practice:
    
    1. **Prohibition of Punitive Actions:**
       - Segment labels and risk indicators must **NEVER** be utilized as automated inputs or justification for termination, demotion, salary reduction, or disciplinary action.
    2. **Descriptive Exploration vs. Deterministic Prediction:**
       - Unsupervised clustering discovers statistical patterns; it does **NOT** predict that a specific employee will resign or fail. Claiming calibrated individual attrition probabilities from clustering is methodologically invalid.
    3. **Exclusion of Sensitive Demographics:**
       - Protected personal attributes (e.g., Gender, Race, Age where applicable) were deliberately excluded from distance calculations to prevent algorithmic bias and disparate impact.
    4. **Human-in-the-Loop Governance:**
       - Machine learning outputs serve solely as decision support for organizational resource planning, leadership development, workload balancing, and employee engagement programs. Final talent decisions must always remain with human HR professionals.
    """)
