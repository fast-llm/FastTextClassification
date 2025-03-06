
export version=v0.2
sudo docker build  --build-arg BUILD_DATE=$(date +%Y-%m-%d:%H:%M:%S) \
    -t classification-worker:${version} \
    -f ./docker/inference/${version}/Dockerfile .