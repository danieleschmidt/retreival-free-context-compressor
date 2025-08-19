#!/bin/bash
# Production Deployment Script for Retrieval-Free Context Compression
set -e

ENVIRONMENT=${1:-production}
NAMESPACE="compression-service"
IMAGE_TAG=${2:-latest}

echo "🚀 Starting deployment to $ENVIRONMENT..."

# Validate prerequisites
echo "🔍 Validating prerequisites..."
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ docker not found. Please install docker."
    exit 1
fi

# Build and push Docker image
echo "🔨 Building Docker image..."
docker build -t terragon/retrieval-free-compression:$IMAGE_TAG .

if [ "$ENVIRONMENT" = "production" ]; then
    echo "📤 Pushing to production registry..."
    docker push terragon/retrieval-free-compression:$IMAGE_TAG
fi

# Apply Kubernetes manifests
echo "📋 Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/pod-security-policy.yaml
kubectl apply -f k8s/network-policy.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/compression-service -n $NAMESPACE

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

# Run health checks
echo "🏥 Running health checks..."
sleep 30  # Allow time for pods to start

EXTERNAL_IP=$(kubectl get service compression-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
if [ "$EXTERNAL_IP" = "localhost" ]; then
    kubectl port-forward service/compression-service 8080:80 -n $NAMESPACE &
    PORT_FORWARD_PID=$!
    sleep 5
    HEALTH_URL="http://localhost:8080/health"
else
    HEALTH_URL="http://$EXTERNAL_IP/health"
fi

if curl -f "$HEALTH_URL" >/dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    exit 1
fi

# Clean up port forward if used
if [ -n "$PORT_FORWARD_PID" ]; then
    kill $PORT_FORWARD_PID 2>/dev/null || true
fi

echo "🎉 Deployment completed successfully!"
echo "📊 Access monitoring at: http://grafana.your-domain.com"
echo "🔍 View logs with: kubectl logs -f -l app=compression-service -n $NAMESPACE"
