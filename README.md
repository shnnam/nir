# Neural Image Representations for Multi-Image Fusion and Layer Separation
### [Project Page](https://shnnam.github.io/research/nir) | [Paper](https://arxiv.org/abs/2108.01199)
A PyTorch implementation of the paper ["Neural Image Representations for Multi-Image Fusion and Layer Separation"](https://arxiv.org/abs/2108.01199).


## Jupyter notebook scripts
- Our code is implemented on Jupyter notebook.
- Download [data](https://drive.google.com/file/d/1_Ft7GOwlUUOg8lp7LB2VMc6ZFQRvokJa/view?usp=sharing), and unzip the file under the root. e.g. `~/nir/data`
- `./visualization.ipynb`  
Visualization of learned canonical view in neural image representations.
- `./moire_removal.ipynb`, `./obstruction_removal.ipynb`, `./rain_removal.ipynb`  
Moire removal, obstruction removal (reflection, fence), rain removal


## Citation
Please cite our paper when you use this code.
```
@inproceedings{nam2022neural,
  title={Neural Image Representations for Multi-Image Fusion and Layer Separation},
  author={Nam, Seonghyeon and Brubaker, Marcus A and Brown, Michael S},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Reference
Our implementation is based on the official code of SIREN: [https://github.com/vsitzmann/siren](https://github.com/vsitzmann/siren)


## Contact
Please contact [snam0331 AT gmail.com](snam0331@gmail.com) if you have any question about this work.