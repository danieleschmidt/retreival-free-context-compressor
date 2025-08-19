#!/bin/bash
# Scaling script for Retrieval-Free Context Compression
set -e

NAMESPACE="compression-service"
REPLICAS=${1:-3}
ACTION=${2:-scale}

echo "📈 Starting scaling operation..."

case $ACTION in
    scale)
        echo "🔧 Scaling to $REPLICAS replicas..."
        kubectl scale deployment compression-service --replicas=$REPLICAS -n $NAMESPACE
        ;;
    autoscale)
        MIN_REPLICAS=${1:-3}
        MAX_REPLICAS=${2:-50}
        TARGET_CPU=${3:-70}
        echo "🤖 Setting up autoscaling ($MIN_REPLICAS-$MAX_REPLICAS replicas, $TARGET_CPU% CPU target)..."
        kubectl autoscale deployment compression-service --min=$MIN_REPLICAS --max=$MAX_REPLICAS --cpu-percent=$TARGET_CPU -n $NAMESPACE
        ;;
    status)
        echo "📊 Current scaling status:"
        kubectl get deployment compression-service -n $NAMESPACE
        kubectl get hpa -n $NAMESPACE
        kubectl get pods -n $NAMESPACE
        exit 0
        ;;
    *)
        echo "❌ Unknown action: $ACTION"
        echo "Usage: $0 [replicas] [scale|autoscale|status]"
        exit 1
        ;;
esac

# Wait for scaling
echo "⏳ Waiting for scaling to complete..."
kubectl rollout status deployment/compression-service -n $NAMESPACE

echo "✅ Scaling completed successfully!"
kubectl get pods -n $NAMESPACE
