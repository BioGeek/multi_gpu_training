build:
	sudo singularity build pytorch.sif pytorch.def

clean:
	find . -name "*.lprof" -type f -delete