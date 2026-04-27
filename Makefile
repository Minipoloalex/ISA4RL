NAME ?= default-name
DIR ?= .

.PHONY: highway metadrive build push clean

highway:
	$(MAKE) build NAME=$(NAME) DIR=src/highway_agents

metadrive:
	$(MAKE) build NAME=$(NAME) DIR=src/metadrive_agents

build:
	docker build -t $(NAME) -f $(DIR)/Dockerfile .
	docker save $(NAME) -o $(NAME).tar
	apptainer build $(NAME).sif docker-archive://$(NAME).tar

push:
	scp $(NAME).sif server_mia:/home/up202108837/$(NAME).sif

clean:
	rm -f *.tar *.sif
