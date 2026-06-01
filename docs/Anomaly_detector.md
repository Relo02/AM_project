## First method
The problem setting is to detect missing scratches from the once provided by the frangi ridge filter.
The assesment is to, therefore, figure out what are the missing scratches and what truly is the noisy pixels that might be considered wrongly by the ridge filter.

### System architecture
let be the input image defined as $I(x, y)$, with $x$ and $y$ the pixels coordinates.
From the Frangi ridge filter, we get the following non-smooth function:

$$
M(x, y) =
\begin{cases}
1  & \text{if} (x, y) \in \mathbb{S} \\  
0  & \text{if} (x, y) \not\in \mathbb{S}
\end{cases}
$$Where $\mathbb{S}$ is the set of scratches for each patch. 
The general objective, is not to learn the "normal glass" structure from pixels very close to the scratches, because scratch borders may contain weak anomalous signal. So what we can do is to dilate in order to make scratch regions thicker. 

 Therefore, the normal candidate is defined as:
 $$
 \mathcal{N} = \{(x, y): M_{dil}(x, y) = 0 \} 
 $$
 This "normal set" does not take into account the boarder pixels, since borders contains artifacts.  

So after we have defined the region of our problem, we define a feature vector for every pixel. Instead of describing each pixel only through its RGB or grayscale intensity, we describe it through a set of handcrafted cues that are useful for separating clean glass from scratch-like structures.

Let the feature vector be defined as:

$$  
f(x, y) \in \mathbb{R}^{K}  
$$

where $K$ is the number of features extracted for each pixel. A possible definition is:

$$  
f(x, y) =  
\begin{bmatrix}  
I_{flat}(x, y) \  
\sigma_{loc}(x, y) \  
|\nabla I(x, y)| \  
F_{Frangi}(x, y) \  
\Delta I(x, y) \  
T_{tophat}(x, y)  
\end{bmatrix}  
$$

where:

- $I_{flat}(x, y)$ is the flat-field corrected intensity;
    
- $\sigma_{loc}(x, y)$ is the local standard deviation computed in a small neighbourhood around the pixel;
    
- $|\nabla I(x, y)|$ is the gradient magnitude;
    
- $F_{Frangi}(x, y)$ is the response of the Frangi ridge filter;
    
- $\Delta I(x, y)$ is the Laplacian response;
    
- $T_{tophat}(x, y)$ is the top-hat response, useful for highlighting small bright or dark structures.
    

The gradient magnitude is defined as:

$$  
|\nabla I(x, y)| =  
\sqrt{  
\left(\frac{\partial I}{\partial x}\right)^2  
+  
\left(\frac{\partial I}{\partial y}\right)^2  
}  
$$

while the Laplacian is defined as:

$$  
\Delta I(x, y) =  
\frac{\partial^2 I}{\partial x^2}  
+  
\frac{\partial^2 I}{\partial y^2}  
$$

The idea is that clean glass pixels should have feature vectors that are relatively similar to each other, while scratches, dust, and other local defects should produce feature vectors that deviate from the normal glass distribution.

Since the majority of pixels in every patch are assumed to be clean glass, we estimate the normal-glass distribution using only the pixels belonging to the set $\mathcal{N}$. Therefore, we collect all feature vectors:

$$  
\mathcal{F}_{\mathcal{N}} =  
{ f(x, y) : (x, y) \in \mathcal{N} }  
$$

and we model them as a multivariate Gaussian distribution:

$$  
f \mid \mathcal{N} \sim \mathcal{N}(\mu, \Sigma)  
$$

where $\mu$ is the mean feature vector of the normal pixels and $\Sigma$ is the covariance matrix.

The mean vector is estimated as:

$$  
\mu =  
\frac{1}{|\mathcal{N}|}  
\sum_{(x, y) \in \mathcal{N}}  
f(x, y)  
$$

and the covariance matrix is estimated as:

$$  
\Sigma =  
\frac{1}{|\mathcal{N}| - 1}  
\sum_{(x, y) \in \mathcal{N}}  
\left(f(x, y) - \mu\right)  
\left(f(x, y) - \mu\right)^T  
$$

However, in practice, the covariance matrix can be unstable, especially when some features are highly correlated. For this reason, a shrinkage version of the covariance matrix can be used:

$$  
\Sigma_{\lambda}

(1 - \lambda)\Sigma  
+  
\lambda \alpha I  
$$

where $I$ is the identity matrix, $\lambda \in [0,1]$ is the shrinkage parameter, and $\alpha$ is usually chosen as:

$$  
\alpha =  
\frac{\text{trace}(\Sigma)}{K}  
$$

Therefore:

$$  
\Sigma_{\lambda}

(1 - \lambda)\Sigma  
+  
\lambda  
\frac{\text{trace}(\Sigma)}{K}  
I  
$$

This regularization makes the covariance matrix easier to invert and makes the anomaly score more stable.

Once the normal-glass model has been estimated, each pixel is scored using the Mahalanobis distance:

$$  
D(x, y) =

\sqrt{  
\left(f(x, y) - \mu\right)^T  
\Sigma_{\lambda}^{-1}  
\left(f(x, y) - \mu\right)  
}  
$$

The Mahalanobis distance measures how far a pixel is from the normal-glass distribution. A low value of $D(x, y)$ means that the pixel is similar to the normal glass pixels used to estimate the model. A high value of $D(x, y)$ means that the pixel is statistically different from the normal glass and can therefore be considered anomalous.

This distance is more suitable than a simple Euclidean distance because it takes into account the variance of each feature and the correlations between different features. For example, if a certain feature naturally changes a lot in clean glass, then a deviation in that feature should not immediately be considered anomalous. On the other hand, if a feature is usually very stable, then even a small deviation may be meaningful.

After computing the anomaly score for every pixel, we define two thresholds using the distribution of $D(x, y)$ over the normal set $\mathcal{N}$:

$$  
\tau_{lo} = q_{95}  
$$

$$  
\tau_{hi} = q_{99.9}  
$$

where $q_{95}$ is the 95th percentile and $q_{99.9}$ is the 99.9th percentile of the anomaly scores computed on the normal pixels:

$$  
{D(x, y) : (x, y) \in \mathcal{N}}  
$$

The lower threshold $\tau_{lo}$ is used to verify whether a pixel already detected by the Frangi mask is also anomalous according to the normal-glass model. The higher threshold $\tau_{hi}$ is used to detect only very strong anomalies that were not detected by the Frangi mask.

At this point, the original Frangi mask $M(x, y)$ and the anomaly score $D(x, y)$ can be fused in order to obtain a three-way decomposition.

The first class is the set of confirmed scratches:

$$  
C(x, y)

M(x, y)  
\wedge  
\left(D(x, y) \geq \tau_{lo}\right)  
$$

These are pixels that were detected by the Frangi ridge filter and are also anomalous according to the normal-glass model. Therefore, they are more reliable than the pixels detected only by the classical pipeline.

The second class is the set of candidate missed scratches:

$$  
A(x, y)

\left(D(x, y) \geq \tau_{hi}\right)  
\wedge  
\neg M(x, y)  
$$

These are pixels that were not detected by the Frangi ridge filter but have a very high anomaly score. This means that they may correspond to scratches missed by the classical pipeline.

However, not every anomalous pixel should be considered a scratch. Some anomalous pixels may correspond to dust, noise, reflections, or small illumination artifacts. For this reason, the candidate missed scratches are passed through the same morphological and component-based filters used in the classical pipeline:

The purpose of this step is to keep only anomalies that have a scratch-like shape. For example, a valid scratch candidate should usually be thin, elongated, and locally coherent in orientation. Small round blobs or isolated noisy pixels should be rejected.

The third class is the set of suspect false positives:

$$  
F(x, y)

M(x, y)  
\wedge  
\left(D(x, y) < \tau_{lo}\right)  
$$

These are pixels that were detected by the Frangi ridge filter but do not look anomalous according to the normal-glass model. Therefore, they may correspond to false positives introduced by the classical pipeline, such as dust particles or background texture wrongly interpreted as scratches.

Finally, the fused scratch mask is obtained by combining the confirmed scratches with the cleaned candidate missed scratches:

$$  
M_{fused}(x, y)

C(x, y)  
\cup  
A_{clean}(x, y)  
$$

This fused mask has two advantages. First, it keeps the most reliable detections from the Frangi ridge filter. Second, it adds possible scratches that were missed by the classical method but are statistically anomalous with respect to the local normal-glass model.

The final output of the system is therefore not only a binary segmentation mask, but a more informative decomposition:

$$  
\text{Output}

{C, A_{clean}, F, M_{fused}}  
$$

where:

- $C$ represents confirmed scratches;
    
- $A_{clean}$ represents candidate missed scratches;
    
- $F$ represents suspect false positives;
    
- $M_{fused}$ represents the final fused scratch mask.
    

This formulation is useful because the Frangi mask is treated as a weak prior rather than as ground truth. The method does not try to train a model to imitate the Frangi output. Instead, it uses the Frangi mask to define a safe normal region and then learns, in an unsupervised way, what normal glass looks like inside each patch.

This is important because the classical mask may be incomplete or noisy. If a supervised model were trained directly on $M(x, y)$, the model would inherit the same errors of the ridge filter. In contrast, this method focuses on the disagreement between the classical mask and the unsupervised anomaly score. This disagreement is the most useful information for improving the scratch detection pipeline.