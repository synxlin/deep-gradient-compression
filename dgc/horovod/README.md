# Horovod Patch

We applied [patch](horovod.june.6.6b77884.patch) to Horovod at [this commit](https://github.com/horovod/horovod/tree/6b77884daf92649ecf031fcc8ff29697bbea0132).
Nonetheless, we copied the modified files in this directory so that you don't have to patch Horovod source code.

The modification is very subtile:

- class `Compressor` will take `name` as another argument when compressing a tensor.

- class `DistributedOptimizer` will perform `communicate()` of its compression member if possible, instead of always using `allreduce_async()`.
