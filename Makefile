# Makefile for IP3G Docker workflows

# Configurable variables
IMAGE ?= ip3g
TAG ?= latest
PLATFORM ?=

DOCKER_BUILD_PLATFORM := $(if $(PLATFORM),--platform=$(PLATFORM),)

.PHONY: help build run run-preprocess bash push

help:
	@echo "Targets:"
	@echo "  make build             - Build Docker image ($(IMAGE):$(TAG))"
	@echo "  make run               - Run visualize_data.py with mounted data/output"
	@echo "  make run-preprocess    - Run preprocess_data.py with mounted data/output"
	@echo "  make bash              - Start a shell inside the container"
	@echo "  make push              - Push image to registry (ensure IMAGE includes registry)"
	@echo "Variables: IMAGE, TAG, PLATFORM (e.g., PLATFORM=linux/amd64)"

build:
	docker build $(DOCKER_BUILD_PLATFORM) -t $(IMAGE):$(TAG) .

run:
	mkdir -p loaded_data images
	docker run --rm -it \
		-v $(PWD)/loaded_data:/app/loaded_data \
		-v $(PWD)/images:/app/images \
		-v $(PWD)/GTEX_phenotype:/app/GTEX_phenotype:ro \
		-v $(PWD)/gtex_RSEM_Hugo_norm_count:/app/gtex_RSEM_Hugo_norm_count:ro \
		-v $(PWD)/gtex_gene_expected_count:/app/gtex_gene_expected_count:ro \
		$(IMAGE):$(TAG)

run-preprocess:
	mkdir -p loaded_data images
	docker run --rm -it \
		-v $(PWD)/loaded_data:/app/loaded_data \
		-v $(PWD)/images:/app/images \
		-v $(PWD)/GTEX_phenotype:/app/GTEX_phenotype:ro \
		-v $(PWD)/gtex_RSEM_Hugo_norm_count:/app/gtex_RSEM_Hugo_norm_count:ro \
		-v $(PWD)/gtex_gene_expected_count:/app/gtex_gene_expected_count:ro \
		$(IMAGE):$(TAG) preprocess_data.py

bash:
	mkdir -p loaded_data images
	docker run --rm -it \
		-v $(PWD)/loaded_data:/app/loaded_data \
		-v $(PWD)/images:/app/images \
		--entrypoint /bin/bash \
		$(IMAGE):$(TAG)

push:
	docker push $(IMAGE):$(TAG)
