# Kubernetes deployment assets

This folder contains a Helm chart for the Creative Ad Generation API and optional manifests for running MinIO and MLflow to mimic the local docker-compose experience.

## API Helm chart

The chart lives in `helm/api` and ships the Deployment, Service, ConfigMap/Secret, Horizontal Pod Autoscaler, and an Ingress with Prometheus annotations enabled by default.

### Usage

```bash
helm upgrade --install creative-ad-api ./helm/api \
  --set image.repository=ghcr.io/your-org/creative-ad-api \
  --set image.tag=$(git rev-parse --short HEAD)
```

Key settings in `helm/api/values.yaml`:

- `config.*`: API host/port, MLflow tracking URI, and model artifact path.
- `secret.*`: AWS-style credentials used by MLflow/MinIO; disable with `secret.create=false` if you want to supply your own secret refs.
- `ingress.*`: host and class configuration for ingress routing.
- `autoscaling.*`: toggle and tune the HPA thresholds.
- `podAnnotations`: includes Prometheus scrape hints (path `/metrics`, port `8000`).

## Optional MinIO + MLflow demo stack

To run a self-contained experiment backend, apply the demo manifests after installing the chart:

```bash
kubectl apply -f k8s/demo/minio.yaml
kubectl apply -f k8s/demo/mlflow.yaml
```

The demo resources create:

- A MinIO deployment with persistent storage, default admin credentials (`minioadmin`/`minioadmin`), and a setup job that provisions the `mlflow` bucket.
- An MLflow server backed by a SQLite database stored on a persistent volume and configured to use the MinIO service for artifact storage.

Point the API to MLflow via `config.mlflowTrackingUri=http://mlflow:5000` and the credentials in the generated `minio-credentials` secret.
