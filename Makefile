build_pytorch:
	sudo singularity build pytorch.sif pytorch.def

build_jax:
	sudo singularity build jax.sif jax.def

clean:
	find . -name "*.lprof" -type f -delete
	find . -name "*.out" -type f -delete