# Controller Benchmark

Bechmarking scripts require the following python packages to be installed:

```
pip install transforms3d
pip install tabulate
```

To use the suite, modify local parameter file `controller_benchmark.yaml` to include desired controller plugin.

Then execute the benchmarking:

- `ros2 launch ./controller_benchmark_bringup.py` to launch part of the nav2 stack.
- `python3 metrics.py --ros-args -p use_sim_time:=true`  to launch the benchmark script
- `python3 ./process_data.py` to take the metric files and process them into key results
