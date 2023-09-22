# Readme

### Overview:
This documentation provides insight into optimizing embedding dimensions to discern 'similarity' and 'relatedness' relations efficiently.

### Similarity:
- Identifying similarity is inherently simpler, requiring fewer dimensions to construct effective embeddings, focusing primarily on achieving higher AUC and facilitating efficient code mapping.

### Relatedness:
- Capturing relatedness is more intricate, necessitating more extensive dimensions to elevate the AUC, particularly when dealing with multiple elements like drug side effects and phenotyping.

### Methodology:
- Unlike approaches such as KERSER, which utilize rotation matrices to handle various relations and thus consume excessive parameters, our method optimizes for self-adjustable parameters.
- It employs the initial dimensions to compute similarity and leverages the entirety of dimensions for relatedness computations, ensuring balanced resource allocation and superior efficiency in modeling diverse relations.

### Objectives:
- Streamline parameter utilization to avoid redundancy.
- Enhance the preciseness and efficiency in determining both similarity and relatedness relations within restrictive dimensional scopes.

### Conclusion:
This optimized method is concise and enables more efficient parameter use, providing an accurate and resource-efficient approach to modeling complex relations.
