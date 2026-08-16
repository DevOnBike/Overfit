@echo off
setlocal

REM ---------------------------------------------------------------------------------------------------
REM Removes everything install.cmd created, and nothing else.
REM
REM Two steps, because Helm deliberately does not do the second one: PersistentVolumeClaims survive a
REM release deletion so an accidental uninstall does not destroy the data. That is the right default for a
REM production cluster and the wrong one for a lab you want gone, so the PVCs are removed explicitly and
REM only within this namespace.
REM
REM The CRDs installed by kube-prometheus-stack (ServiceMonitor, PrometheusRule, ...) are cluster-scoped
REM and are left in place: removing them would break any other Prometheus operator install sharing this
REM cluster. They are inert on their own.
REM ---------------------------------------------------------------------------------------------------

echo === removing the Helm release ===
helm uninstall overfit-lab --namespace monitoring

echo.
echo === removing the volumes Helm leaves behind ===
kubectl delete pvc --all -n monitoring

echo.
echo === removing the namespace ===
kubectl delete ns monitoring

echo.
echo Done. CRDs from the chart are intentionally left in place — see the comment in this script.
echo Nothing outside the `monitoring` namespace was touched.
endlocal
