# GBFS external scheduler fallback

GitHub's native `schedule` event remained inactive even though the repository is public, the workflow is active on the default branch, Actions are enabled, and manual dispatch succeeds. The collector and GitHub Release persistence are retained unchanged.

The lowest-permission practical fallback is a free external HTTPS scheduler such as cron-job.org calling GitHub's workflow-dispatch API every 15 minutes:

```text
POST https://api.github.com/repos/zhangtingning146-oss/citibike-demand-forecast/actions/workflows/schedule-gbfs.yml/dispatches
Authorization: Bearer <fine-grained token stored only in the scheduler secret store>
Accept: application/vnd.github+json
Content-Type: application/json

{"ref":"master"}
```

The fine-grained token should be restricted to this one public repository with only **Actions: read and write** permission and a short expiration. It must never be committed, placed in a URL query string, saved in project files, or pasted into a report/chat. Configure the external service's secret header field directly in its authenticated web interface.

Verification requires two external automatic runs roughly 15 minutes apart, both showing persisted Release growth and zero duplicate `(feed_last_updated_utc, station_id)` keys. Runs triggered manually or by a local computer do not count.

This fallback cannot be activated by the project code alone: the repository owner must create the external scheduler account and token, then store the token in that service. Until that authorization is completed, the status remains “collector deployed and manually verified; automatic scheduling unverified.”
