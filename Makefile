# Default name if not provided via command line
NAME ?= example-name

.PHONY: build push clean

build:
	docker build -t $(NAME) .
	docker save $(NAME) -o $(NAME).tar
	apptainer build $(NAME).sif docker-archive://$(NAME).tar

push:
	scp $(NAME).sif server_mia:/home/up202108837/$(NAME).sif

clean:
	rm -f $(NAME).tar $(NAME).sif
