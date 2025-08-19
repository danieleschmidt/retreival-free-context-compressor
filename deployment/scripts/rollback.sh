#!/bin/bash
# Rollback script for Retrieval-Free Context Compression
set -e

NAMESPACE="compression-service"
PREVIOUS_VERSION=${1:-previous}

echo "🔄 Starting rollback to $PREVIOUS_VERSION..."

# Get current deployment status
echo "📊 Current deployment status:"
kubectl get deployment compression-service -n $NAMESPACE

# Perform rollback
echo "⏪ Rolling back deployment..."
kubectl rollout undo deployment/compression-service -n $NAMESPACE

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/compression-service -n $NAMESPACE

# Verify rollback
echo "✅ Verifying rollback..."
kubectl get pods -n $NAMESPACE

# Run health checks
echo "🏥 Running health checks..."
sleep 30

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
    echo "✅ Rollback health check passed!"
else
    echo "❌ Rollback health check failed!"
    exit 1
fi

# Clean up port forward if used
if [ -n "$PORT_FORWARD_PID" ]; then
    kill $PORT_FORWARD_PID 2>/dev/null || true
fi

echo "🎉 Rollback completed successfully!"
