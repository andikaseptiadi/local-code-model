# Interactive Tutorials

This directory contains Jupyter notebooks that provide interactive, hands-on tutorials for understanding and experimenting with the transformer implementation.

## Setup

### Prerequisites

1. **Install Jupyter**:
   ```bash
   pip install jupyter
   ```

2. **Install gophernotes** (Go kernel for Jupyter):
   ```bash
   # Install gophernotes
   go install github.com/gopherdata/gophernotes@latest

   # Create kernel directory
   mkdir -p ~/Library/Jupyter/kernels/gophernotes

   # Install kernel spec (adjust path to your GOPATH if needed)
   cp $(go env GOPATH)/pkg/mod/github.com/gopherdata/gophernotes@*/kernel/* ~/Library/Jupyter/kernels/gophernotes/
   ```

   Or follow the official installation guide: https://github.com/gopherdata/gophernotes

3. **Verify installation**:
   ```bash
   jupyter kernelspec list
   ```

   You should see `gophernotes` in the list.

### Running Notebooks

1. **Start Jupyter**:
   ```bash
   cd local-code-model
   jupyter notebook
   ```

2. **Open a notebook**: Navigate to `notebooks/` and open any `.ipynb` file

3. **Select Go kernel**: When prompted, select "Go" or "gophernotes" as the kernel

## Notebooks

### 01-tensor-basics.ipynb
**Level**: Beginner
**Duration**: 30-45 minutes

Learn the foundation of all neural network operations:
- Creating and manipulating tensors
- Matrix operations (MatMul, transpose, reshape)
- Activation functions (ReLU, GELU, Softmax)
- Broadcasting and element-wise operations
- Hands-on exercises

**Prerequisites**: Basic Go knowledge

### 02-attention-mechanism.ipynb
**Level**: Intermediate
**Duration**: 45-60 minutes
**Status**: Coming soon

Build the attention mechanism from scratch:
- Query, Key, Value projections
- Scaled dot-product attention
- Multi-head attention
- Visualizing attention patterns
- Hands-on: implement your own attention layer

**Prerequisites**: Complete notebook 01

### 03-training-transformer.ipynb
**Level**: Advanced
**Duration**: 60-90 minutes
**Status**: Coming soon

Train a complete transformer model:
- Building a full transformer architecture
- Training loop with backpropagation
- Optimizer (Adam) and learning rate scheduling
- Loss curves and evaluation
- Hands-on: train on your own dataset

**Prerequisites**: Complete notebooks 01 and 02

## Tips

### Running Code Cells

- **Execute cell**: `Shift + Enter`
- **Execute and stay**: `Ctrl + Enter`
- **Insert cell below**: `B` (in command mode)
- **Delete cell**: `D, D` (press D twice in command mode)

### Common Issues

**Issue**: `cannot find package`
**Solution**: Make sure you're running Jupyter from the project root directory

**Issue**: Kernel crashes
**Solution**: Restart kernel (`Kernel → Restart`) and re-run cells from the top

**Issue**: Import errors
**Solution**: Ensure the Go module path matches your project:
```go
import "github.com/scttfrdmn/local-code-model"
```

### Best Practices

1. **Run cells in order**: Notebooks are designed to be executed sequentially
2. **Experiment freely**: Modify code cells and see what happens
3. **Save often**: Use `Cmd/Ctrl + S` to save your work
4. **Restart when confused**: `Kernel → Restart & Clear Output` gives a clean slate

## Additional Resources

- **Code Reference**: See the main codebase in the parent directory
- **Deep Dive Docs**: Check `../docs/` for detailed explanations:
  - `attention-mechanism.md`
  - `backpropagation.md`
  - `training-dynamics.md`
- **Learning Guide**: See `../LEARNING.md` for comprehensive learning path

## Contributing

Found an issue or want to add a notebook? Contributions welcome! Please:
1. Keep the educational focus
2. Include clear explanations
3. Provide hands-on exercises
4. Test all code cells before submitting

## Feedback

Questions or suggestions? Open an issue on GitHub!
