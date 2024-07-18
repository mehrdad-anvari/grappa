# Fast Implementation of GRAPPA Algorithm
Welcome to the repository for the fast implementation of the GRAPPA algorithm!
This project leverages the GPU capabilities offered by the PyTorch library to significantly accelerate the GRAPPA algorithm,
making it particularly useful for training deep neural networks (DNNs) to reconstruct undersampled multi-coil MRI data.
The GRAPPA algorithm is commonly used as a preprocessing step in MRI reconstruction, and our enhancements can provide a substantial speedup.

## Implementation Details
Our implementation is inspired by the pygrappa repository, with the following key modifications:
1. PyTorch Integration: We have replaced Numpy with PyTorch to leverage GPU acceleration.
2. Matrix Operations: Instead of using for loops, we utilize matrix operations to improve computational efficiency.
3. Faster Undersampling Pattern Detection: We have incorporated techniques to find unique undersampling patterns more rapidly.

In typical scenarios, this optimized code can achieve speedups of up to 30 times compared to standard implementations.

## Contributing
We welcome contributions to improve this implementation further. If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## Acknowledgements
We would like to thank the authors of the pygrappa repository for their inspiring work, which served as the foundation for this project.
