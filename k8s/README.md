# Kubernetes Dask

1. Create a cluster:

```
openstack coe cluster create --template kubernetes ...
```

2. Build Image

```
docker build -t lukasheinrich/atlas100daskworker -f k8s/image/Dockerfile .
docker push lukasheinrich/atlas100daskworker
```

3. Install Dask

```
helm repo add dask https://helm.dask.org/
helm repo update
helm install my-dask dask/dask -f k8s/config.yml
```


4. Upgrading

In case you want to change the Helm config upgrade via

```
helm upgrade my-dask dask/dask -f k8s/config.yml
```
