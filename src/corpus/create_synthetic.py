#!/usr/bin/env python3
"""
Create synthetic documents for RAG corpus
Generates FAQs, policies, and technical docs for controlled testing
"""

import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Document Templates
# Each category represents a realistic document type found in
# real-world RAG deployments (company knowledge bases, helpdesks)
# ─────────────────────────────────────────────────────────────

FAQ_DOCS = [
    {
        "title": "How to Reset Your Password",
        "text": (
            "FAQ: How do I reset my password?\n\n"
            "To reset your password, navigate to the login page and click 'Forgot Password'. "
            "Enter your registered email address and you will receive a password reset link within 5 minutes. "
            "The reset link expires after 24 hours for security reasons. "
            "If you do not receive the email, check your spam folder or contact support at support@example.com. "
            "For security, never share your password with anyone, including support staff."
        )
    },
    {
        "title": "How to Create an Account",
        "text": (
            "FAQ: How do I create a new account?\n\n"
            "To create an account, visit our registration page and fill in your full name, email address, "
            "and a strong password. Your password must be at least 12 characters and contain uppercase letters, "
            "lowercase letters, numbers, and special characters. "
            "After submitting the form, you will receive a verification email. "
            "Click the link in that email to activate your account. "
            "Account activation links expire after 48 hours."
        )
    },
    {
        "title": "How to Update Billing Information",
        "text": (
            "FAQ: How do I update my billing information?\n\n"
            "To update your billing information, log in to your account and navigate to Settings > Billing. "
            "Click 'Update Payment Method' and enter your new credit card details. "
            "We accept Visa, Mastercard, American Express, and PayPal. "
            "All payment information is encrypted using AES-256 and stored securely. "
            "Changes to billing information take effect immediately for future charges. "
            "For refund requests, contact billing@example.com within 30 days of purchase."
        )
    },
    {
        "title": "How to Cancel Subscription",
        "text": (
            "FAQ: How do I cancel my subscription?\n\n"
            "To cancel your subscription, log in and go to Settings > Subscription > Cancel Plan. "
            "Your access will continue until the end of the current billing period. "
            "We do not offer prorated refunds for mid-cycle cancellations. "
            "After cancellation, your data is retained for 30 days before permanent deletion. "
            "You can reactivate your account within this 30-day window without losing your data. "
            "To delete your data immediately, submit a data deletion request through Settings > Privacy."
        )
    },
    {
        "title": "System Requirements",
        "text": (
            "FAQ: What are the system requirements?\n\n"
            "Our platform supports the following configurations: "
            "Web browsers: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+. "
            "Operating systems: Windows 10+, macOS 11+, Ubuntu 20.04+. "
            "Minimum hardware: 4GB RAM, 2GHz dual-core processor, 1280x720 display resolution. "
            "Internet connection: minimum 5 Mbps for standard use, 25 Mbps for video features. "
            "JavaScript must be enabled. Cookies must be allowed for authentication to work."
        )
    },
    {
        "title": "How to Export Your Data",
        "text": (
            "FAQ: How do I export my data?\n\n"
            "You can export all your data at any time from Settings > Privacy > Export Data. "
            "The export includes your profile information, activity history, documents, and settings. "
            "Data is exported as a ZIP archive containing JSON and CSV files. "
            "Large exports may take up to 24 hours to prepare. "
            "You will receive an email notification when your export is ready to download. "
            "Export files are available for download for 7 days before they are deleted from our servers."
        )
    },
    {
        "title": "Two-Factor Authentication Setup",
        "text": (
            "FAQ: How do I set up two-factor authentication?\n\n"
            "Two-factor authentication (2FA) adds an extra layer of security to your account. "
            "To enable 2FA, go to Settings > Security > Two-Factor Authentication and click Enable. "
            "We support authenticator apps (Google Authenticator, Authy) and SMS verification. "
            "Authenticator apps are recommended over SMS for stronger security. "
            "After enabling 2FA, you will be prompted for a verification code at each login. "
            "Save your backup codes in a secure location in case you lose access to your authenticator device."
        )
    },
    {
        "title": "Contacting Support",
        "text": (
            "FAQ: How do I contact customer support?\n\n"
            "Our support team is available through multiple channels. "
            "Live chat: available 24/7 through the in-app chat widget. "
            "Email: support@example.com with a response time of 24 hours on business days. "
            "Phone: 1-800-SUPPORT, available Monday to Friday, 9 AM to 6 PM EST. "
            "For urgent security issues, use security@example.com for a faster response. "
            "When contacting support, include your account email and a description of the issue "
            "to help us resolve your request more quickly."
        )
    },
]

POLICY_DOCS = [
    {
        "title": "Security Policy",
        "text": (
            "Information Security Policy\n\n"
            "All users and employees must adhere to the following security guidelines. "
            "Passwords must be at least 12 characters and include uppercase letters, lowercase letters, "
            "numbers, and special symbols. Passwords must be changed every 90 days. "
            "Never reuse the last 5 passwords. "
            "Two-factor authentication is mandatory for all accounts with administrative privileges. "
            "Users must lock their workstations when away from their desks. "
            "Suspicious emails must be reported to security@example.com immediately. "
            "Violation of this policy may result in account suspension or termination of employment."
        )
    },
    {
        "title": "Data Privacy Policy",
        "text": (
            "Data Privacy and Protection Policy\n\n"
            "We are committed to protecting the privacy of our users. "
            "We collect only the personal data necessary to provide our services, "
            "including name, email address, and usage data. "
            "We do not sell or share personal data with third parties for marketing purposes. "
            "All data is encrypted in transit using TLS 1.3 and at rest using AES-256. "
            "Users have the right to access, correct, and delete their personal data at any time. "
            "We comply with GDPR, CCPA, and other applicable data protection regulations. "
            "Data breaches will be reported to affected users and relevant authorities within 72 hours."
        )
    },
    {
        "title": "Acceptable Use Policy",
        "text": (
            "Acceptable Use Policy\n\n"
            "Users of our platform agree not to engage in any of the following prohibited activities. "
            "Attempting to gain unauthorized access to other accounts or systems. "
            "Distributing malware, spyware, or any harmful software. "
            "Harassing, threatening, or intimidating other users. "
            "Posting content that infringes on intellectual property rights. "
            "Using the service for any illegal purpose under applicable laws. "
            "Attempting to overload or disrupt the platform's infrastructure. "
            "Violations may result in immediate account termination and referral to law enforcement."
        )
    },
    {
        "title": "Remote Work Policy",
        "text": (
            "Remote Work Security Policy\n\n"
            "Employees working remotely must follow these security requirements. "
            "Use only company-approved devices for accessing work systems. "
            "Connect through the corporate VPN at all times when accessing internal resources. "
            "Do not use public Wi-Fi without VPN protection. "
            "Ensure your home network router uses WPA3 encryption. "
            "Keep all software and operating systems updated with the latest security patches. "
            "Do not allow family members or others to use work devices. "
            "Report lost or stolen devices to IT security immediately."
        )
    },
    {
        "title": "Data Retention Policy",
        "text": (
            "Data Retention and Deletion Policy\n\n"
            "This policy defines how long different types of data are retained. "
            "User account data is retained for the duration of the account plus 30 days after deletion. "
            "Transaction records are retained for 7 years to comply with financial regulations. "
            "Server access logs are retained for 90 days for security monitoring purposes. "
            "Marketing email records are retained for 2 years. "
            "Support ticket records are retained for 3 years. "
            "After retention periods expire, data is securely deleted using industry-standard methods. "
            "Users may request early deletion of their data subject to legal requirements."
        )
    },
    {
        "title": "Incident Response Policy",
        "text": (
            "Security Incident Response Policy\n\n"
            "This policy outlines the procedure for responding to security incidents. "
            "All suspected security incidents must be reported to the security team within 1 hour of discovery. "
            "The incident response team will assess severity within 2 hours: Critical, High, Medium, or Low. "
            "Critical incidents trigger immediate escalation to senior management and legal counsel. "
            "Affected systems may be isolated to prevent further damage. "
            "Evidence must be preserved and documented for forensic analysis. "
            "Post-incident reviews are conducted within 5 business days to prevent recurrence. "
            "Security incidents involving personal data trigger GDPR notification procedures."
        )
    },
]

TECHNICAL_DOCS = [
    {
        "title": "API Authentication Guide",
        "text": (
            "API Authentication Documentation\n\n"
            "All API requests must be authenticated using Bearer tokens. "
            "Include your API key in the Authorization header: 'Authorization: Bearer YOUR_API_KEY'. "
            "API keys can be generated in the developer dashboard under Settings > API Keys. "
            "Each API key has configurable permissions: read-only, read-write, or admin. "
            "Rate limits apply: Free tier allows 100 requests per minute, "
            "Pro tier allows 1000 requests per minute, Enterprise tier is configurable. "
            "Exceeding rate limits returns HTTP 429 with a Retry-After header. "
            "API keys should be stored securely and never committed to version control."
        )
    },
    {
        "title": "Database Configuration Guide",
        "text": (
            "Database Configuration Documentation\n\n"
            "The system uses PostgreSQL 14 as the primary database. "
            "Connection string format: postgresql://username:password@host:5432/database_name. "
            "Configure connection pooling with a maximum of 100 connections and a 30-second timeout. "
            "Enable SSL mode for all production database connections. "
            "Backup schedule: full backup daily at 2:00 AM UTC, incremental backups every 6 hours. "
            "Database migrations are managed using Alembic. Run 'alembic upgrade head' to apply migrations. "
            "Never run migrations directly against production without testing in staging first."
        )
    },
    {
        "title": "Deployment Guide",
        "text": (
            "Application Deployment Guide\n\n"
            "The application is deployed using Docker containers orchestrated by Kubernetes. "
            "Build the Docker image: 'docker build -t app:latest .' "
            "Push to container registry: 'docker push registry.example.com/app:latest'. "
            "Apply Kubernetes manifests: 'kubectl apply -f k8s/'. "
            "Environment variables are managed through Kubernetes Secrets. "
            "Horizontal pod autoscaling is configured to scale between 2 and 10 replicas based on CPU usage. "
            "Health check endpoint: GET /api/health returns 200 OK when the service is ready. "
            "Rolling deployments ensure zero downtime during updates."
        )
    },
    {
        "title": "Monitoring and Alerting Setup",
        "text": (
            "Monitoring and Alerting Documentation\n\n"
            "Application monitoring is implemented using Prometheus and Grafana. "
            "Metrics are collected every 15 seconds and retained for 30 days. "
            "Key metrics monitored: request latency, error rate, throughput, and resource utilization. "
            "Alerts are triggered when: error rate exceeds 1% for 5 minutes, "
            "response latency p99 exceeds 2 seconds for 5 minutes, "
            "or CPU usage exceeds 80% for 10 minutes. "
            "Alerts are sent to PagerDuty for on-call engineers and Slack for team awareness. "
            "Runbooks for common alerts are maintained in the internal wiki."
        )
    },
    {
        "title": "Error Codes Reference",
        "text": (
            "API Error Codes Reference\n\n"
            "The API uses standard HTTP status codes with detailed error messages. "
            "400 Bad Request: The request is malformed or missing required parameters. "
            "401 Unauthorized: Authentication is missing or the API key is invalid. "
            "403 Forbidden: The authenticated user lacks permission for the requested action. "
            "404 Not Found: The requested resource does not exist. "
            "409 Conflict: The request conflicts with the current state of the resource. "
            "429 Too Many Requests: The rate limit has been exceeded. Check the Retry-After header. "
            "500 Internal Server Error: An unexpected server error occurred. Contact support if it persists. "
            "All error responses include a JSON body with 'code', 'message', and 'request_id' fields."
        )
    },
    {
        "title": "Webhook Configuration Guide",
        "text": (
            "Webhook Configuration Documentation\n\n"
            "Webhooks allow your application to receive real-time notifications about events. "
            "Register a webhook endpoint in Settings > Developer > Webhooks. "
            "Your endpoint must respond with HTTP 200 within 10 seconds to confirm receipt. "
            "Failed deliveries are retried up to 5 times with exponential backoff. "
            "Webhook payloads are signed using HMAC-SHA256. "
            "Verify the signature by computing HMAC-SHA256 of the raw request body using your webhook secret "
            "and comparing it to the X-Signature-256 header. "
            "Always verify signatures before processing webhook payloads to prevent replay attacks."
        )
    },
    {
        "title": "Caching Strategy Guide",
        "text": (
            "Application Caching Documentation\n\n"
            "The application uses Redis for caching frequently accessed data. "
            "Cache keys follow the naming convention: 'entity_type:entity_id:field'. "
            "Default time-to-live (TTL) values: user sessions 24 hours, API responses 5 minutes, "
            "database query results 1 minute, static content 7 days. "
            "Cache invalidation is event-driven: when data is updated, related cache entries are deleted. "
            "Use cache-aside pattern: check cache first, if miss fetch from database and populate cache. "
            "Monitor cache hit rate using the /metrics endpoint. A healthy hit rate is above 80%. "
            "Avoid caching sensitive personal data unless it is encrypted."
        )
    },
    {
        "title": "Search Feature Documentation",
        "text": (
            "Search Functionality Documentation\n\n"
            "The platform provides full-text search powered by Elasticsearch. "
            "Search queries support boolean operators: AND, OR, NOT. "
            "Use quotes for exact phrase matching: 'machine learning'. "
            "Wildcard searches are supported: 'machi*' matches 'machine' and 'machinery'. "
            "Results are ranked by relevance score combining text match quality and recency. "
            "Search results are paginated with a maximum of 100 results per page. "
            "Filters can be applied for date range, category, and document type. "
            "Search analytics are available in the admin dashboard under Reports > Search."
        )
    },
]


def create_synthetic(num_docs=50):
    """
    Create synthetic documents

    Args:
        num_docs: Total number of documents to create (default: 50)
    """
    print("="*70)
    print("Synthetic Document Generator")
    print("="*70)
    print(f"\nTarget: {num_docs} synthetic documents")
    print("Categories: FAQs, Policies, Technical Docs")
    print("="*70)

    # Create output directory
    output_dir = Path("data/raw/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate how many documents per category
    # Distribute evenly across 3 categories
    per_category = num_docs // 3
    extra = num_docs % 3  # distribute remainder to first categories

    faq_count    = per_category + (1 if extra > 0 else 0)
    policy_count = per_category + (1 if extra > 1 else 0)
    tech_count   = per_category

    print(f"\n[1/3] Building document list...")
    print(f"      FAQs:            {faq_count}")
    print(f"      Policies:        {policy_count}")
    print(f"      Technical Docs:  {tech_count}")

    # Build document list
    documents = []
    doc_id = 1

    # Add FAQs
    for i in range(faq_count):
        template = FAQ_DOCS[i % len(FAQ_DOCS)]
        documents.append({
            "id": f"synthetic_{doc_id}",
            "title": template["title"],
            "text": template["text"],
            "category": "faq",
            "source": "synthetic",
            "trust_score": 0.5
        })
        doc_id += 1

    # Add Policies
    for i in range(policy_count):
        template = POLICY_DOCS[i % len(POLICY_DOCS)]
        documents.append({
            "id": f"synthetic_{doc_id}",
            "title": template["title"],
            "text": template["text"],
            "category": "policy",
            "source": "synthetic",
            "trust_score": 0.5
        })
        doc_id += 1

    # Add Technical Docs
    for i in range(tech_count):
        template = TECHNICAL_DOCS[i % len(TECHNICAL_DOCS)]
        documents.append({
            "id": f"synthetic_{doc_id}",
            "title": template["title"],
            "text": template["text"],
            "category": "technical",
            "source": "synthetic",
            "trust_score": 0.5
        })
        doc_id += 1

    print(f"\n[2/3] Generated {len(documents)} documents")

    # Save to JSON
    print(f"\n[3/3] Saving to file...")

    output_file = output_dir / f"synthetic_{num_docs}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"      ✓ Saved to: {output_file}")
    print(f"      ✓ File size: {file_size_mb:.2f} MB")

    # Print summary
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nCreated: {len(documents)} synthetic documents")
    print(f"Output file: {output_file}")

    # Category breakdown
    print("\nBreakdown by category:")
    for cat in ["faq", "policy", "technical"]:
        count = sum(1 for d in documents if d["category"] == cat)
        print(f"  {cat:12s}: {count} documents")

    # Show samples
    print("\n" + "-"*70)
    print("Sample Documents (one per category):")
    print("-"*70)

    shown = set()
    for doc in documents:
        if doc["category"] not in shown:
            print(f"\n[{doc['category'].upper()}] {doc['title']}")
            print(f"    Text preview: {doc['text'][:120]}...")
            shown.add(doc["category"])
        if len(shown) == 3:
            break

    print("\n" + "="*70)
    print("Next step: python src/corpus/combine_corpus.py")
    print("="*70)

    return documents


if __name__ == "__main__":
    # Configuration
    NUM_DOCS = 50  # Change this to create more/less synthetic documents

    print("\n⚠️  IMPORTANT:")
    print(f"   This will create {NUM_DOCS} synthetic documents")
    print("   No internet connection required")
    print("   Estimated time: < 1 minute")
    print()

    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("\n❌ Cancelled by user")
        exit(0)

    try:
        documents = create_synthetic(num_docs=NUM_DOCS)

        if documents:
            print("\n✅ Synthetic document creation complete!")
            exit(0)
        else:
            print("\n❌ Creation failed")
            exit(1)

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
