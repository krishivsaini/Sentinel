variable "project_id" {
  description = "The GCP project to deploy into."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run and Artifact Registry."
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  description = "owner/name of the repository allowed to deploy (via OIDC)."
  type        = string
  default     = "krishivsaini/Sentinel"
}

variable "memory" {
  description = "Memory per instance: PyTorch, two local models and the indexes."
  type        = string
  default     = "2Gi"
}

variable "daily_spend_limit_usd" {
  description = "The API stops answering for the day once its metered LLM spend reaches this."
  type        = number
  default     = 0.05
}

variable "rate_limit" {
  description = "Limit on /query per named account, or per IP for the demo account (slowapi syntax; several limits separated by ';')."
  type        = string
  default     = "6/minute;60/day"
}

variable "cors_origins" {
  description = "Browser origins allowed to call the API (the deployed dashboard)."
  type        = list(string)
  default     = []
}

variable "billing_account" {
  description = "Billing account ID for the budget alert (XXXXXX-XXXXXX-XXXXXX); empty to skip it."
  type        = string
  default     = ""
}

variable "monthly_budget" {
  description = "Monthly amount for the budget alert, in budget_currency."
  type        = number
  default     = 5
}

variable "budget_currency" {
  description = "Currency of the billing account (the alert must match it)."
  type        = string
  default     = "USD"
}

variable "redis_cache" {
  description = "Keep the semantic cache in Redis (the rediss:// URL in Secret Manager secret sentinel-redis-url) instead of instance memory, so cache hits survive restarts and scale-to-zero. Add a secret version before turning this on; see deploy/README.md."
  type        = bool
  default     = false
}

variable "named_accounts" {
  description = "Mount a users file (Secret Manager secret sentinel-users) so named accounts with their own scopes can sign in. Add a secret version before turning this on; see deploy/README.md."
  type        = bool
  default     = false
}
