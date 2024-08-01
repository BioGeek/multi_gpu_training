import os
import jax
import jax.numpy as jnp

jax.distributed.initialize()
# jax.config.update("jax_enable_x64", True)

print(f"CUDA VISIBLE DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"Total devices: {jax.device_count()}")
print(f"Devices per task: {jax.local_device_count()}")

x = jnp.ones(jax.local_device_count())

# Computes a reduction (sum) across all devices of x
# and broadcast the result, in y, to all devices.
# If x=[1] on all devices and we have 32 devices,
# the result is y=[32] on all devices.

y = jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(x)

print(y)                                                       