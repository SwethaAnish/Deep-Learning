## SELECTIVE SEARCH FOR OBJECT DETECTION

### Felzenszwalb’s Algorithm - An image segmentation algorithm

1. Weight of an edge is non-negative measure of the dissimilarity between the two pixels connected by that edge.
2. Graph will be constructed either as Grid graph(surrounding 8 pixels) or as a Nearest neighbour graph (Each pixel is a point in the feature space (x, y, r, g, b), in which (x, y) is the pixel location and (r, g, b) is the color values in RGB
3. Based on the weight values of edges the segmentation will be performed
4. Larger the number of edges considered(k), finer the segmentation would be, if smaller number of edges are considered, the segmentation would be coarser.

Selective search Algorithm works based on Felzenszwalb’s Algorithm.

#### Key Take aways:
1. Selective search algorithm oversegments an image based on color similarity, texture similarity size, shape similarity and meta similarity (linear combination of above). General idea is that a region proposal algorithm should inspect the image and attempt to find the regions in the image that likely contain an object.
2. The "mode" parameter in selective search algorithm refers to the different combinations of strategies. The strategies is found here (https://medium.com/dataseries/understanding-selective-search-for-object-detection-3f38709067d7)
3. It is mostly used in Region based Covolutional Neural Networks (R-CNN)

#### References:

1. https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/
2. https://www.analyticsvidhya.com/blog/2021/05/image-segmentation-with-felzenszwalbs-algorithm
