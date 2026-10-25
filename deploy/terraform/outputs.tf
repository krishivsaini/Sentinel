output "service_url" {
  description = "The API's public URL."
  value       = google_cloud_run_v2_service.api.uri
}

output "image_repository" {
  description = "Where GitHub Actions pushes images."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

# The three values below go into the repository's Actions variables (not secrets: none is secret).
output "github_variables" {
  description = "Set these as GitHub Actions repository variables."
  value = {
    GCP_PROJECT_ID      = var.project_id
    GCP_REGION          = var.region
    GCP_WIF_PROVIDER    = google_iam_workload_identity_pool_provider.github.name
    GCP_DEPLOYER_SA     = google_service_account.deployer.email
  }
}

output "secrets_to_fill" {
  description = "Add a value to each with: printf '%s' \"$VALUE\" | gcloud secrets versions add NAME --data-file=-"
  value       = [for s in google_secret_manager_secret.app : s.secret_id]
}
