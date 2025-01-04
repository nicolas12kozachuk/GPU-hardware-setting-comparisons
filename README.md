# Performance Comparison of Different GPU Hardware Settings

## Project Description

This project evaluates the performance of various **GPU hardware settings** using CUDA programming. It explores three configurations: 
1. **Single GPU with a Single Stream** 
2. **Single GPU with Concurrent Streams** 
3. **Multiple GPUs**

The performance is compared across three computational tasks:
1. **Vector Element Doubling**
2. **Vector Addition**
3. **2D Matrix Multiplication**

The experiments were conducted using Lehigh University’s HPC (High-Performance Computing) cluster with NVIDIA GPUs and CUDA tools.

## Key Findings

- **Concurrent Streams**: Demonstrated significant performance improvements due to parallel execution within a single GPU.
- **Multiple GPUs**: Delivered the highest performance gains for large-scale computations by distributing tasks across multiple isolated GPUs.
- **Task-Dependent Performance**:
  - Tasks with minimal data dependencies, such as vector operations, showed the greatest speedups.
  - Tasks with complex data dependencies, such as matrix multiplication, had more limited improvements due to memory management overhead.

## Technical Details

- **Hardware**: NVIDIA GeForce GPUs (12GB) using CUDA 12.0.
- **Code Implementation**:
  - Written in CUDA C/C++.
  - Profiling conducted with NVIDIA tools.
- **Tasks**:
  - Vector operations (element doubling, addition) and matrix multiplication.
- **Performance Analysis**:
  - Profiling metrics include execution time, memory transfers, and kernel execution time.

### Results Summary
| Configuration              | Task                  | Execution Time | Speedup     |
|----------------------------|-----------------------|----------------|-------------|
| Single GPU, Single Stream  | Vector Addition       | 13,606,168 ns | Baseline    |
| Single GPU, Concurrent     | Vector Addition       | 3,301,670 ns  | ~4x         |
| Multiple GPUs              | Vector Addition       | 113,857 ns    | ~120x       |

*Note: Performance varies depending on task complexity and data dependencies.*

## Code Structure

1. **Single GPU with Single Stream**:
   - Basic CUDA implementation with serial execution.
2. **Single GPU with Concurrent Streams**:
   - Utilizes multiple streams for parallelism.
3. **Multiple GPUs**:
   - Distributes workload across multiple GPUs with memory management.

## How to Run

1. Access the HPC system via SSH:
ssh -XY username@ece-hpc0?.cc.lehigh.edu

2. Load the CUDA environment:
source /path/to/cuda-environment-script.sh scl enable devtoolset-7 bash

3. Compile and run the CUDA code:
nvcc -o executable_name program_name.cu -run

4. Profile the execution:
nsys profile --stats=true -o report_name executable_name


## Learning Outcomes

This project provided key insights into:
- **GPU Architecture**: Understanding single and multi-GPU configurations and their parallelism capabilities.
- **Performance Optimization**: Leveraging CUDA features like concurrent streams for enhanced efficiency.
- **Memory Management**: Addressing challenges with data dependencies and memory transfers.

## References

1. NVIDIA CUDA Programming Model
2. NVIDIA Profiler Tool Documentation
3. Relevant CUDA Courses: [Accelerating CUDA Applications](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-01+V1)

---

For more details, refer to the project report.



