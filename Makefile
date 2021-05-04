PROJECT = kinetics-downloader
WORKSPACE = /workspace/$(PROJECT)
DOCKER_IMAGE = $(PROJECT):latest

DOCKER_LOGIN := "eval $$\( aws ecr get-login --registry-ids 929292782238 --no-include-email --region us-east-1 \)"
# BASE_DOCKER_IMAGE ?= nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# BASE_DOCKER_IMAGE ?= 929292782238.dkr.ecr.us-east-1.amazonaws.com/ouroboros:master-latest
# BASE_DOCKER_IMAGE ?= 929292782238.dkr.ecr.us-east-1.amazonaws.com/ouroboros-evaluate:master-latest
BASE_DOCKER_IMAGE ?= 929292782238.dkr.ecr.us-east-1.amazonaws.com/ouroboros-evaluate:master-21

DOCKER_OPTS = \
	-it \
	--rm \
	-e DISPLAY=${DISPLAY} \
	-v /data:/data \
	-v /tmp:/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /mnt/fsx:/mnt/fsx \
	-v /root/.ssh:/root/.ssh \
	-v ~/.aws:/root/.aws \
	-v $(HOME)/.ouroboros:/root/.ouroboros \
	--shm-size=1G \
	--ipc=host \
	--network=host \
	--privileged \
	-e DATASETS_ROOT=/mnt/fsx/datasets/
	# -e PYTHONPATH=/workspace/detectron2/../:/workspace/ouroboros/:/workspace/packnet-sfm:/home/tri_ouroboros_evaluate/:/workspace/cityscapesScripts:/workspace/fvcore/:/workspace/tridet/ \
	# -v $(HOME)/projects/fvcore:/workspace/fvcore

DOCKER_BUILD_ARGS = \
	--build-arg WORKSPACE=$(WORKSPACE) \
	--build-arg AWS_ACCESS_KEY_ID \
	--build-arg AWS_SECRET_ACCESS_KEY \
	--build-arg AWS_DEFAULT_REGION \
	--build-arg WANDB_ENTITY \
	--build-arg WANDB_API_KEY \
	--build-arg BASE_DOCKER_IMAGE=$(BASE_DOCKER_IMAGE)

docker-login:
	@eval $(DOCKER_LOGIN)

docker-pull-base: docker-login
	docker pull ${BASE_DOCKER_IMAGE}

docker-build: docker-login
	docker build \
	$(DOCKER_BUILD_ARGS) \
	-f ./Dockerfile \
	-t $(DOCKER_IMAGE) .

docker-dev:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	-v $(PWD):$(WORKSPACE) \
	$(DOCKER_IMAGE) bash

docker-start:
	nvidia-docker run --name $(PROJECT) \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

clean:
	find . -name '"*.pyc' | xargs rm -f && \
	find . -name '__pycache__' | xargs rm -rf

dist-run:
	nvidia-docker run --name $(PROJECT) --rm \
		-e DISPLAY=${DISPLAY} \
		-v ~/.torch:/root/.torch \
		${DOCKER_OPTS} \
		-v $(PWD):$(WORKSPACE) \
		${DOCKER_IMAGE} \
		${COMMAND}

docker-run:
	nvidia-docker run --name $(PROJECT) --rm \
		-e DISPLAY=${DISPLAY} \
		-v ~/.torch:/root/.torch \
		${DOCKER_OPTS} \
		${DOCKER_IMAGE} \
		${COMMAND}

dist-sweep:
	nvidia-docker run --name $(PROJECT) --rm \
        -e DISPLAY=${DISPLAY} \
        -e HOST=${HOST} \
        -e WORLD_SIZE=${WORLD_SIZE} \
        -e WANDB_PROJECT=${WANDB_PROJECT} \
        -e WANDB_ENTITY=${WANDB_ENTITY} \
        -e WANDB_API_KEY=${WANDB_API_KEY} \
        -e WANDB_AGENT_REPORT_INTERVAL=300 \
        -v ~/.torch:/root/.torch \
        ${DOCKER_OPTS} \
		-v $(PWD):$(WORKSPACE) \
        ${DOCKER_IMAGE} \
        ${COMMAND}
