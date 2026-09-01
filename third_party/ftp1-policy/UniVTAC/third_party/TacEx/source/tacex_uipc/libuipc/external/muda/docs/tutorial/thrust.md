# Thrust

[Thrust](https://github.com/NVIDIA/thrust) is a C++ parallel programming library developed by Nvidia. It provides a unified and convenient parallel algorithm call similar to the C++ STL API. Currently, thrust, along with `cub` and `libcudacxx`, has been included in cccl (CUDA C++ Core Libraries), becoming the most core official library of cuda c++ in essence.

The most important contribution of Thrust is to provide a set of iterator interfaces to replace pointers. Thrust's iterators not only carry the information of the original pointer, but also carry the information of the execution backend. A concrete example is that we can use `type_traits` to distinguish whether the memory pointed to by the iterator belongs to the device or host.

Based on the iterator interface, Thrust provides functions similar to the standard library `<algorithm>`, such as `thrust::sort`, `thrust::reduce`, etc. These algorithms are generally called `parallel primitives` and are the cornerstone of parallel programming.

Thrust belongs to the high-level API, and we generally use iterators to use its algorithms. But there is always a moment when we need to manually design some kernels to achieve the desired functionality. At this time, we may have to use raw pointers to read memory.

Your code may look like this:
```c++
#include <thrust/device_vector.h>
void pure_thrust()
{
    using namespace thrust;
    // Execute on the default stream and synchronize after the function call
    auto nosync_policy = thrust::cuda::par_nosync.on(nullptr);

    constexpr auto N = 1000;
    
    // Apply for device side vector
    device_vector<int> buffer(N);
    
    // parallel for
    // Use count_iterator begin=0, end=N, to construct a sequence from 0 to N
    // This sequence will be passed into i when calling the lambda expression
    // So we get a similar effect to parallel_for
    // We don't need sync, because the device_vector will synchronize when destructed.
    for_each(nosync_policy,
             make_counting_iterator(0),
             make_counting_iterator(N), 
             [buffer = buffer.data()] __device__(int i) mutable
             {
                 buffer[i] = i;
             });
}
```

Here comes the problem, the access to `buffer[i]` is unsafe. As the problem becomes more complex, especially when encountering access like `buffer[map[i]]`, it is difficult to avoid out-of-bounds or null pointer problems.

Luckily, MUDA provides a solution to this problem. 

We can modify the code like this:
```c++
#include <muda/muda.h>
void muda_thrust()
{
    using namespace muda;
    using namespace thrust;

    auto nosync_policy = thrust::cuda::par_nosync.on(nullptr);

    constexpr auto N = 1000;

    device_vector<int> buffer(N);

    {
        // Label the Kernel Name to get clearer debug output.
        KernelLabel label{__FUNCTION__};

        for_each(nosync_policy,
                 make_counting_iterator(0),
                 make_counting_iterator(N),
                 [
                  // Create a muda::Dense1D viewer from device_vector to safely access memory.
                  buffer = Dense1D<int>(raw_pointer_cast(buffer.data()), N).name("buffer")
                 ] __device__(int i) mutable
                 {
                     buffer(i + 1) = i;
                 });
    }
}
```

Note that the code here has been slightly modified, and we intentionally made the buffer write out of bounds.

We will get the following output:
```
(1|2)-(231|256):<error> Dense1D[buffer:muda_thrust]: out of range, index=(1000) m_dim=(1000)
```

We will see that muda correctly reports the out-of-bounds object as buffer in `muda_thrust` kernel, because the accessed index is greater than or equal to the container size.

It is a bit cumbersome to construct `Dense1D` every time, so muda provides a container `DeviceVector` that inherits from `device_vector`. The code can be rewritten as:

```c++
void muda_thrust()
{
    using namespace muda;
    using namespace thrust;

    auto nosync_policy = thrust::cuda::par_nosync.on(nullptr);

    constexpr auto N = 1000;

    DeviceVector<int> buffer(N);

    {
        KernelLabel label{__FUNCTION__};
        for_each(nosync_policy,
                 make_counting_iterator(0),
                 make_counting_iterator(N),
                 [
                  // Create a viewer from DeviceVector to safely access memory.
                  buffer = buffer.viewer().name("buffer")
                 ] __device__(int i) mutable
                 {
                     buffer(i + 1) = i;
                 });
    }
}
```