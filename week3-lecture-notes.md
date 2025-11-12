# Week 3 - Unsupervised Learning

UCL AI Society  
November 2025  

## 1. Last Week Recap

Last week, we talked about **Linear Regression**.  
Our goal was to find a parameter vector $\mathbf{w}$ that best describes a linear relationship between some input $\mathbf{x}$ and an output $y$:

$$
y = \mathbf{w}^\top \mathbf{x} + \epsilon
$$

where $\epsilon$ is some noise term, often normally distributed: $\epsilon \sim \mathcal{N}(0, \sigma)$.

- y is the outcome or target (for example, the price of a house)  
- **x** is the input vector of features (for example, containing the number of rooms, size, location, etc.)  
- **w** are the parameters of our model, i.e. what we want to estimate.

We introduced **Ordinary Least Squares (OLS)**, which minimizes the squared distance between predictions $\hat{y} = \mathbf{w}^\top \mathbf{x}$ and the true values $y$:

$$
  L(\mathbf{w}) = \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i)^2
$$

In vector form:

$$
L(\mathbf{w}) = \| \mathbf{y} - X\mathbf{w} \|^2
$$

The matrix $X$ is simply a table grouping all of our data points $x_i$. So the first row in $X$ is our first point $x_1$, etc...

The goal is to find the vector of parameters $\mathbf{w}$ that minimizes that loss:

$$
\hat{\mathbf{w}}_{\text{OLS}} = \arg \max_{\mathbf{w}} L(\mathbf{w})
$$

To minimize this loss, we differentiate it with respect to $\mathbf{w}$ and set the gradient to zero. We obtain the **normal equations**:

$$
\mathbf{w}_{\text{OLS}} = (X^\top X)^{-1} X^\top \mathbf{y}
$$

We saw that $X^\top X$ can be non-invertible if features are correlated.  
To fix this, we added a small penalty on large weights, giving **Ridge Regression**:

$$
L_\lambda(\mathbf{w}) = \| \mathbf{y} - X\mathbf{w} \|_2^2 + \lambda \|\mathbf{w}\|_2^2
$$

and the new solution:

$$
\mathbf{w}_{\text{OLS},\lambda} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}
$$

That was all from a *geometric* or *optimization* perspective.  
Now we'll see how all of this can also be derived **probabilistically**.

## 2. Maximum Likelihood Estimation (MLE)

Let's now look at linear regression through the lens of probability.

We'll assume that each $y_i$ is drawn from a probability distribution, given $x_i$.  
Specifically, let's assume:

$$
y_i = \mathbf{w}^\top \mathbf{x}_i + \epsilon_i \quad \text{with} \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Then, conditioned on $x_i$, we have:

$$
y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^\top \mathbf{x}_i, \sigma^2)
$$

We can now write the probability of observing all our data points $(x_i, y_i)$:

$$
L(\mathbf{w}) = p(y_1, ..., y_N \mid x_1, ..., x_N, \mathbf{w}) = \prod_{i=1}^N p(y_i \mid x_i, \mathbf{w})
$$

The product appears because we assume our samples are independent.

Each term is a Gaussian probability density:

$$
p(y_i \mid x_i, \mathbf{w}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(y_i - \mathbf{w}^\top x_i)^2}{2\sigma^2}\right]
$$

Putting it all together:

$$
L(\mathbf{w}) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(y_i - \mathbf{w}^\top x_i)^2}{2\sigma^2}\right]
$$

This is called the **likelihood function** - it tells us how probable the data is, given parameters **w**. Our goal is to maximize the likelihood, so find the parameter **w** that makes it the most likely to see our data.

Maximizing this product directly is messy, so we usually take the log:

$$
\ell(\mathbf{w}) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \mathbf{w}^\top x_i)^2
$$

This works becomes the logarithm is monotonally increasing. The first term is constant with respect to **w**, so maximizing $\ell(w)$ is equivalent to minimizing the sum of squared errors.

$$
\text{maximize } \ell(\mathbf{w}) \quad \Longleftrightarrow \quad \text{minimize } \sum_i (y_i - \mathbf{w}^\top x_i)^2
$$

This means: **OLS is simply Maximum Likelihood Estimation under Gaussian noise.**

## 3. Bayesian Statistics and MAP Estimation

In MLE, **w** is treated as fixed, we just find the one that best explains the data.

In **Bayesian statistics**, we treat **w** as a random variable itself, and express our uncertainty about it through a *prior* distribution $p(w)$. This represents what we think of $w$ before having seen the data.

For example, we could assume $\mathbf{w} \sim \mathcal{N}(0, \tau)$. This means that before looking our data, we assume $\mathbf{w}$ is around 0, with a ~60% confidence of it being in the range $[-\tau, \tau]$.

After seeing data, we update this belief using Bayes' rule:

$$
p(w \mid X, y) = \frac{p(y \mid X, w) \, p(w)}{p(y \mid X)}
$$

In plain words, this means

$$
\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}
$$

Note that the evidence $p(y \mid X)$ does not depend on **w**, it is just a normalization constant. While the prior represents our uncertainty in $\mathbf{w}$ **before** seing the data, the posterior represents our uncertainty in  $\mathbf{w}$ **after** seing the data.

We typically want to select the most probable **w** after observing data, so maximize the posterior distribution. This is the Maximum A Posteriori (MAP) estimate:

$$
w_{\text{MAP}} = \arg\max_w p(w \mid X, y)
$$

Again, for simplicity, we take the log of the posterior:

$$
\begin{aligned}
\log p(\mathbf{w} \mid X, y)
&= \log p(y \mid X, \mathbf{w}) + \log p(\mathbf{w}) - \log p(y \mid X)\\
&= \Big[-\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i)^2\Big] + \Big[-\frac{d}{2}\log(2\pi\tau^2) - \frac{1}{2\tau^2}\sum^N_{i=1}w_i^2\Big] + \text{const.}\\
&= -\frac{1}{2\sigma^2}\|y - X\mathbf{w}\|^2 - \frac{1}{2\tau^2}\|\mathbf{w}\|^2 + \text{const.}
\end{aligned}
$$

So maximizing the posterior is equivalent to minimizing:

$$
\|y - Xw\|^2 + \lambda \|w\|^2 \quad \text{where } \lambda = \frac{\sigma^2}{\tau^2}
$$

That's exactly **Ridge Regression** again! So now we can see **Ridge as a MAP estimator** with a Gaussian prior on weights.

## 4. Unsupervised Learning Introduction

Until now, all our models have been **supervised**: we had pairs $(x, y)$.  
In **unsupervised learning**, there are no labels.  
We only have inputs $x_1, x_2, ..., x_N$, and we want to **find structure or patterns** within them.

### Example: Estimating the Mean of a Gaussian

Suppose we have data points $(x_1, \ldots, x_N)$ drawn independently from a Gaussian distribution  
with unknown mean $\mu$ and known variance $\sigma^2$:

$$
x_i \sim \mathcal{N}(\mu, \sigma^2)
$$

We want to find the **Maximum Likelihood Estimate (MLE)** of $\mu$.

The likelihood of observing all data points is:

$$
L(\mu) = p(x_1, \ldots, x_N \mid \mu)
       = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}
         \exp\left[-\frac{(x_i - \mu)^2}{2\sigma^2}\right]
$$

Taking the logarithm of the likelihood (the **log-likelihood**) gives:

$$
\ell(\mu) = \log L(\mu)
          = -\frac{N}{2}\log(2\pi\sigma^2)
            - \frac{1}{2\sigma^2}\sum_{i=1}^N (x_i - \mu)^2
$$

We maximize $\ell(\mu)$ by differentiating with respect to $\mu$ and setting it to zero:

$$
\frac{d\ell}{d\mu}
= -\frac{1}{2\sigma^2} \cdot 2 \sum_{i=1}^N (x_i - \mu)(-1)
= \frac{1}{\sigma^2} \sum_{i=1}^N (x_i - \mu)
$$

Setting this derivative equal to zero:

$$
\sum_{i=1}^N (x_i - \mu) = 0
\quad \Rightarrow \quad
N\mu = \sum_{i=1}^N x_i
$$

Hence, the Maximum Likelihood Estimate is:

$$
\hat{\mu}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N x_i
$$

Therefore, the MLE for the mean of a Gaussian is simply the **sample mean**.

## 5. Coursework Solution

### Minimizing the L1 Loss

Suppose we have data points $(x_1, x_2, \ldots, x_N)$, and we want to find a single value $\mu$ that best represents them by minimizing the **L1 loss**:

$$
L(\mu) = \sum_{i=1}^N |x_i - \mu|
$$

Because of the absolute value, we treat the two cases separately:
- if $x_i < \mu$, then $|x_i - \mu| = \mu - x_i$
- if $x_i > \mu$, then $|x_i - \mu| = x_i - \mu$

We can rewrite the total loss as:

$$
L(\mu) = \sum_{x_i < \mu} (\mu - x_i) + \sum_{x_i > \mu} (x_i - \mu)
$$

Now take the derivative with respect to $\mu$:

$$
\frac{dL}{d\mu} = \sum_{x_i < \mu} 1 - \sum_{x_i > \mu} 1
$$

At the minimum, this derivative must be zero:

$$
\sum_{x_i < \mu} 1 = \sum_{x_i > \mu} 1
$$

This means there are equally many points on either side of $\mu$.  
So the value of $\mu$ that minimizes the L1 loss is the **median** of the data.

$$
\hat{\mu}_{L1} = \text{median}(x_1, \ldots, x_N)
$$

Just like minimizing the L2 loss corresponds to the **mean** under a Gaussian distribution, minimizing the **L1 loss** corresponds to the **median** under a Laplace distribution.  

So:
- Minimizing **L2 loss** gives the **mean**  
- Minimizing **L1 loss** gives the **median**

## 6. Principal Component Analysis (PCA)

### Motivation

Many datasets have features that are correlated or redundant. For example, in biology, flower measurements like petal length and petal width tend to increase together. They are correlated.
We often want to represent data with fewer dimensions while keeping as much of its structure as possible.

That's what **Principal Component Analysis (PCA)** does. It finds new axes, **principal components**, that capture the directions of maximum variance in the data.

By projecting data onto those directions, we can:
- visualize it in fewer dimensions (e.g., 2D or 3D),
- compress it without losing much information,
- remove redundancy between correlated features.

The drawback is that our projection loses interpretability, our features are more abstract and not as simple as "petal width", "petal length", etc.

### Eigenvalues and Eigenvectors Recap

Given a matrix $A$, an **eigenvector** $v$ and **eigenvalue** $\lambda$ satisfy:
$$
A v = \lambda v
$$

This means multiplying by $A$ only *scales* $v$, but does not its direction.  

### PCA

So PCA finds the axes in which the data varies the most. These axes are called **principal components**. To find those, we compute the **empirical covariance matrix**:

$$
S = \frac{1}{N} \sum_{i=1}^N \tilde{x}_i \tilde{x}_i^\top
$$

The covariance matrix $S$ tells us how each pair of features varies together:
- $S_{jj}$ (the diagonal) represents the variance of feature $j$,
- $S_{jk}$ (the off-diagonals) shows how correlated features $j$ and $k$ are.

The **principal components** are the **eigenvectors** of this covariance matrix.  
Each eigenvector $u_k$ defines a new axis in feature space, and its corresponding eigenvalue $\lambda_k$ tells us how much variance the data has along that axis.

We order the eigenvectors by descending eigenvalues:
$$
\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D
$$

The first eigenvector (the one with the largest eigenvalue) is the **first principal component** - the direction in which the data varies the most. The second principal component is orthogonal to the first and captures the next highest variance, and so on.

Intuitively:
- PCA finds the directions where the data "spreads out" the most.
- By projecting onto those directions, we retain most of the variation in the data using fewer dimensions.

This gives us a lower-dimensional representation of the data - while preserving as much variance as possible.

### When is PCA useful

PCA is effective only when the **eigenspectrum**(the list of eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_D$) of the empirical covariance shows a clear drop-off:
$$
\lambda_1 \gg \lambda_2 \gg \dots
$$
This means most of the variance is captured by just a few components.

If all eigenvalues are similar, the data's variance is spread evenly across all directions, and PCA won't help much - there's no low-dimensional structure to exploit.

### Example: The Iris Dataset

Let's apply PCA to the **Iris dataset**, one of the most classic datasets in machine learning.

It contains 150 samples of iris flowers.  
Each sample has **4 measurements**:
1. Sepal length  
2. Sepal width  
3. Petal length  
4. Petal width  

There are **3 species** (Setosa, Versicolor, Virginica). Here, the goal of PCA will be to project the data from 4D to 2D so we can visualize it.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Run PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA projection of the Iris dataset')
plt.show()
```

You'll see that:
- Iris Setosa forms a distinct cluster,
- Versicolor and Virginica overlap slightly.

This shows that PCA found directions that separate the data fairly well, even without using the labels. However, we've lost interpretability. We don't really know what those axis represent anymore.

Now that we've seen the four species clearly form clusters. How do we predict to which specie a flower belongs to?

## 7. K-Means Clustering

### Setup and goal

We observe points $x_1,\dots,x_N \in \mathbb{R}^d$. We believe those points belong to $K$ clusters.  
We assume those $K$ clusters are normal distributions. This forms a **Gaussian Mixture Model (GMM)**.  
A **mixture of normal distributions** is a probabilistic model that assumes the data is generated from a combination of several normal distributions.

$$
p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \mu_k, \Sigma_k)
$$

There are $K$ Gaussian distributions, each with a mean $\mu_k$ and covariance ${\Sigma}_k$.  
$\pi_k$ is the **mixing weight** of cluster $k$. In short, there is a $\pi_k$ chance that a random point $\mathbf{x}$ comes from cluster $k$.

We assume each point in the data has been generated as follows:
1. Randomly pick a cluster label $z$ from $\{1, \dots, K\}$ according to the probabilities $\boldsymbol{\pi} = (\pi_k)_{k=1}^K$.  
   For example, if $z=3$, that means the point was generated by the third Gaussian.
2. Then, sample $\mathbf{x}$ from that cluster's Gaussian: $\mathbf{x} \mid (Z = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$. If $z=3$, we sample $\mathbf{x}$ from a Gaussian with mean $\mu_3$ and covariance $\Sigma_3$.

The variable $z_i$ is the unobserved choice of component $k$ that generated $x_i$. It explains which cluster a point comes from, but we don't actually observe it. We call $z_i$ a **latent variable**.

### From GMM to K-Means

K-Means can be seen as a *simplified version* of this Gaussian mixture model.

We make two strong assumptions:
1. All covariances are spherical and identical, i.e. $\Sigma_k = \sigma^2 I$ for all $k$.
2. All clusters are equally likely, i.e. $\pi_k = \frac{1}{K}$.

Under these assumptions, each Gaussian has the same shape and size. The only thing that differs is the mean $\boldsymbol{\mu}_k$.

### Conceptually

K-Means alternates between two simple ideas:
1. **Assign each point to the cluster with the nearest center.** This finds which cluster each point "belongs" to.  Think of it as estimating the latent variable $z_i$. Conceptually, the closest a point is to the center of a cluster, the more likely it is that it belongs to it.
2. **Move each cluster center to the mean of the points assigned to it.** This updates our estimate of the cluster parameters ($\mu_k$).

We repeat these two steps until nothing changes anymore. Each step reduces the total distance between points and their assigned cluster centers, so the algorithm always converges (though possibly to a local minimum).

### K-Means Algorithm (Pseudo Code)

```text
1. Initialize cluster centers randomly
2. Repeat until convergence:
    # --- Assignment step ---
    For each point x:
        Compute distances to each center.
        Assign x to nearest cluster.

    # --- Update step ---
    For each cluster k:
        Collect all points assigned to cluster k.
        Update the cluster center to the middle / mean of those points.
```

### Example: Clustering the Iris Dataset

There are 3 true species in the Iris dataset (*Setosa*, *Versicolor*, *Virginica*). Let's cluster it with $K=3$:

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
X = load_iris().data

# Run K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize first two features
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-Means Clustering on Iris Data')
plt.show()
```

You'll see three distinct clusters corresponding roughly to the three flower species. K-Means didn't need labels - it discovered structure in the data on its own.

### Connection to the EM Algorithm

K-Means can be viewed as a **special case** of Expectation-Maximization algorithm:
- The E-step is assigning each point to its nearest cluster center.
- The M-step is updating each mean to the average of its assigned points.

It has several simplifying assumptions:
- The covariances are fixed and identical: $\Sigma_k = \sigma^2 I$
- The priors are equal: $\pi_k = \frac{1}{K}$
- The E-step uses **hard assignments**: either you belong to a cluster, or you don't.
The General EM algorithm does not make those assumptions. The math gets complicated...
