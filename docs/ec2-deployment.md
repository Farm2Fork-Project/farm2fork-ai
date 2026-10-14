# EC2 deployment

Merge a tested pull request into **main** to deploy. The workflow runs tests, publishes an immutable `deploy-FULL_SHA` release, then waits for EC2 to confirm activation. PR workflows cannot publish releases. Runtime secrets and model weights stay on EC2; no production `.env` is needed in Actions.

The EC2 deployment agent polls main approximately every minute. Docker manages process restarts. A host lock serializes changes across backend, AI and blockchain. Failed releases are visible in the Actions activation job.

The full operational and recovery runbook is [backend deployment documentation](https://github.com/Farm2Fork-Project/farm2fork-backend/blob/main/docs/ec2-deployment.md). Use `sudo f2fctl status`, `check`, `logs`, `backup`, or `retry COMPONENT` on EC2.

Trained checkpoints are mounted separately from `/opt/farm2fork/models/best_model.pth`. Missing weights keep quality grading unavailable rather than returning random production predictions. Price estimates remain labelled rule-based.

For the frozen grade-only MVP artifact, selected-crop/Other contract, compatible migration order
and model rollback, see [mvp-grading.md](mvp-grading.md). Activate new service code before swapping
the legacy conditioned weights. Weight files are a separate runtime release from CI application images.
