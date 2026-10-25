# Sentinel API on Cloud Run — everything the service needs, as code.
#
#   Artifact Registry    the container images (only the newest few are kept)
#   Secret Manager       GROQ_API_KEY and the JWT secret; Terraform creates the secrets but never
#                        holds their values, so they stay out of state — add them with gcloud
#   Service accounts     one the service runs as (reads its two secrets, nothing else) and one
#                        GitHub Actions deploys as (pushes images, rolls out revisions)
#   Workload Identity    GitHub's OIDC token is exchanged for the deployer's identity, restricted
#                        to this repository's main branch — no long-lived key exists anywhere
#   Cloud Run            scales to zero, one instance at most, CPU billed only during requests
#   Budget alert         optional: emails at 50/90/100% of a monthly amount
#
# The app enforces its own JWT auth, per-IP rate limit and a daily LLM spend cap; Cloud Run's
# public invoker binding only lets requests reach it.

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "this" {}

locals {
  services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ]
  secrets = {
    groq_api_key = "sentinel-groq-api-key"
    jwt_secret   = "sentinel-jwt-secret"
  }
}

resource "google_project_service" "apis" {
  for_each           = toset(concat(local.services, var.billing_account == "" ? [] : ["billingbudgets.googleapis.com"]))
  service            = each.value
  disable_on_destroy = false
}

# --------------------------------------------------------------------------- images

resource "google_artifact_registry_repository" "images" {
  repository_id = "sentinel"
  format        = "DOCKER"
  location      = var.region
  description   = "Sentinel API images"

  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-newest-3"
    action = "KEEP"
    most_recent_versions {
      keep_count = 3
    }
  }
  cleanup_policies {
    id     = "delete-the-rest"
    action = "DELETE"
    condition {
      tag_state = "ANY"
    }
  }
  depends_on = [google_project_service.apis]
}

# --------------------------------------------------------------------------- secrets

resource "google_secret_manager_secret" "app" {
  for_each  = local.secrets
  secret_id = each.value
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# --------------------------------------------------------------------------- identities

resource "google_service_account" "runtime" {
  account_id   = "sentinel-run"
  display_name = "Sentinel API (Cloud Run runtime)"
  depends_on   = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "runtime_reads" {
  for_each  = google_secret_manager_secret.app
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_service_account" "deployer" {
  account_id   = "sentinel-deployer"
  display_name = "Sentinel deploys from GitHub Actions"
  depends_on   = [google_project_service.apis]
}

resource "google_artifact_registry_repository_iam_member" "deployer_pushes" {
  repository = google_artifact_registry_repository.images.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_cloud_run_v2_service_iam_member" "deployer_rolls_out" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.deployer.email}"
}

# Deploying a revision that runs as the runtime account requires acting as it.
resource "google_service_account_iam_member" "deployer_acts_as_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  # Only this repository's main branch can exchange a token at all.
  attribute_condition = "assertion.repository == '${var.github_repository}' && assertion.ref == 'refs/heads/main'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_impersonates_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# --------------------------------------------------------------------------- the service

resource "google_cloud_run_v2_service" "api" {
  name                = "sentinel-api"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.runtime.email
    timeout                          = "120s"
    max_instance_request_concurrency = 4

    scaling {
      min_instance_count = 0 # scale to zero: no cost while idle
      max_instance_count = 1 # one instance, so the in-app daily spend cap is the whole cap
    }

    containers {
      # A placeholder until the first deploy; GitHub Actions rolls out real images after that.
      image = "us-docker.pkg.dev/cloudrun/container/hello"

      resources {
        limits = {
          cpu    = "2"
          memory = var.memory
        }
        cpu_idle          = true # CPU billed only while a request is being handled
        startup_cpu_boost = true # faster model loading on a cold start
      }

      env {
        name  = "SENTINEL_DAILY_SPEND_LIMIT_USD"
        value = tostring(var.daily_spend_limit_usd)
      }
      env {
        name  = "SENTINEL_RATE_LIMIT"
        value = var.rate_limit
      }
      env {
        name  = "SENTINEL_CORS_ORIGINS"
        value = jsonencode(var.cors_origins)
      }
      env {
        name = "GROQ_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.app["groq_api_key"].secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "SENTINEL_JWT_SECRET"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.app["jwt_secret"].secret_id
            version = "latest"
          }
        }
      }

      # The models and indexes load before the server accepts connections (~30-60 s cold).
      startup_probe {
        tcp_socket {
          port = 8080
        }
        period_seconds    = 10
        failure_threshold = 24
      }
      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds = 30
      }
    }
  }

  lifecycle {
    # Images are rolled out by GitHub Actions; Terraform owns everything else.
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }

  depends_on = [google_secret_manager_secret_iam_member.runtime_reads]
}

# The app checks its own JWTs; this only lets requests reach it.
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# --------------------------------------------------------------------------- budget alert

resource "google_billing_budget" "monthly" {
  count           = var.billing_account == "" ? 0 : 1
  billing_account = var.billing_account
  display_name    = "Sentinel monthly budget"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      currency_code = var.budget_currency
      units         = tostring(var.monthly_budget)
    }
  }
  dynamic "threshold_rules" {
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }
  depends_on = [google_project_service.apis]
}
